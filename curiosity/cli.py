#!/usr/bin/env python3
"""Run Curiosity inference from your laptop, over your own SSH login to Curiosity.

Every command opens (or reuses, for 30 minutes) one SSH connection to a login node, uploads this
checkout's serve/, install/ and diagnostics/ to ~/.curiosity-inference/releases/<hash>/ if Curiosity
does not have that version yet, and runs serve/team.py there. Nothing needs to be installed on the
laptop or on Curiosity besides Python (3.7+ here, the system python3 there).

Settings (environment variables or options before the command):
  CURIOSITY_USER   your CINECA login (--user)
  CURIOSITY_BASE   the team folder on Curiosity, e.g. '$WORK/team_vllm' in single quotes (--base)
  CURIOSITY_HOST   the login node you use or were assigned (--host, default login01-ext.curiosity.cineca.it)

Typical first run, from the repository root:
  ./curiosity/cli.py init --account <Slurm project account>
  ./curiosity/cli.py install qwen06b
  ./curiosity/cli.py up qwen06b --qos boost_qos_dbg --time 00:30:00
  ./curiosity/cli.py router up && ./curiosity/cli.py forward

Run ./curiosity/cli.py <command> --help for each command's options and the files it runs.
curiosity/docs/slurm-jobs.md describes the Slurm jobs and how to submit them without this CLI.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

CURIOSITY = Path(__file__).resolve().parent
sys.path.insert(0, str(CURIOSITY / 'tools'))
from update_manifest import build  # noqa: E402

STATE = '"$HOME/.curiosity-inference"'

# What each command runs on Curiosity; shown in --help and in the README.
COMMANDS = {
    'init': ('create the team folder and its settings file',
             'serve/team.py init: creates runtimes/, checkpoints/, endpoints/, users/, logs/, tools/ and\n'
             'inference-site.json (from serve/site.example.json) in the team folder.'),
    'share': ('give teammates access to the team folder, or take it away',
              'serve/team.py share: setfacl on the team folder (POSIX ACLs per login), except other\n'
              "users' personal folders users/<login>/, which stay private."),
    'install': ('download the runtimes and model weights that profiles need',
                'serve/team.py install submits CPU-queue jobs (that queue has internet):\n'
                '  install/runtime.sbatch     image from serve/catalog.json, overlay packages, install/register_runtime.py\n'
                '  install/checkpoint.sbatch  weights from Hugging Face, as pinned in serve/profiles/<profile>.json\n'
                '  install/link_client.sbatch chisel client for the cluster link (--link-client)\n'
                'Logs: <team folder>/logs/.'),
    'up': ('start a model server',
           'serve/team.py up -> serve/inference.py prepare (checks assets, creates a private run directory\n'
           'users/<login>/runs/<profile>-*/ with a fresh API key) and submit -> serve/serve_model.sbatch, which\n'
           'runs serve/node.py (vLLM in the container, one per node) and serve/lifecycle.py (API checks from\n'
           'serve/test_endpoint.py, then the registry record via serve/registry.py).'),
    'status': ('show installs, your servers, your router and the team registry',
               'serve/team.py status: reads squeue, your run directories and <team folder>/endpoints/.'),
    'logs': ('show the log of a job, of a profile\'s newest run, or of the router',
             'serve/team.py logs: tail of the newest matching *.out file.'),
    'down': ('stop model servers (by profile, served name or job id) or the router',
             'serve/team.py down: scancel.'),
    'router': ('start or stop your router, print its token, or store the cluster link password',
               'serve/team.py router -> serve/router.py: copies the code to users/<login>/router/ and submits\n'
               'serve/router.sbatch on the CPU queue; the job runs serve/router.py serve (the HTTP router, and\n'
               'the chisel client if the cluster link is enabled) and queues its successor (router.py next).'),
    'forward': ('forward a port on this laptop to your router (an SSH port forward)',
                'Only reads serve/team.py status. Then runs ssh -N -L 127.0.0.1:<port>:<router node>:<router port>\n'
                'through the login node, so http://127.0.0.1:<port>/v1 on this laptop reaches your router.'),
    'diagnose': ('submit a check job: probe (GPU node), reach (CPU queue -> GPU node, internet), nccl',
                 'serve/team.py diagnose -> diagnostics/probe.sbatch, diagnostics/reach.sbatch or\n'
                 'diagnostics/nccl_check.sbatch (with nccl_task.sh and nccl_check.py). Logs: users/<login>/logs/.'),
    'stage': ('one long 8-GPU B300 reservation: load/unload models + the router inside',
              'serve/team.py stage reserve -> serve/stage_model.sbatch (1 node / 8 B300, long wall-clock), which runs\\n'
              'serve/stage.py serve: owns the rootless docker (diagnostic recipe) and loads/unloads the requested\\n'
              'models on their own 4-GPU slices while running the reused router (router.py serve). load/unload\\n'
              'drop command files that the supervisor inside the reservation polls and acts on; forward reaches the\\n'
              'stage router over SSH like the normal router.'),
    'sync': ('only upload this checkout to Curiosity', 'Nothing runs; prints where the release is.'),
}


CONTROL_PATH = os.path.expanduser('~/.ssh/curiosity-%C')  # one shared, authenticated connection


def ssh_command(args):
    return ['ssh', '-o', 'ControlMaster=auto', '-o', 'ControlPath=' + CONTROL_PATH, '-o', 'ControlPersist=30m',
            '-o', 'ServerAliveInterval=30', args.user + '@' + args.host]


def forward(args, spec):
    """Add a port forward to the shared SSH connection until Ctrl-C, then remove it again.

    Forwards added to a shared (ControlMaster) connection outlive the command that added them, so this
    removes the forward explicitly, and fails instead of opening a second connection with a new login.
    """
    control, target = ssh_command(args)[:-1], ssh_command(args)[-1]
    subprocess.run(control + ['-O', 'cancel', '-L', spec, target], stderr=subprocess.DEVNULL)  # a leftover of ours
    if subprocess.run(control + ['-O', 'forward', '-L', spec, target]).returncode:
        port = spec.split(':')[1]
        raise SystemExit('Could not listen on 127.0.0.1:{0}. Another program uses it (see: lsof -iTCP:{0} -sTCP:LISTEN);\n'
                         'pick another port with --local-port, or close the shared SSH connection, which drops its\n'
                         'forwards: ssh -O exit -o ControlPath={1} {2}'.format(port, CONTROL_PATH, target))
    try:
        while subprocess.run(control + ['-O', 'check', target], stderr=subprocess.DEVNULL).returncode == 0:
            time.sleep(5)
        print('The SSH connection to Curiosity closed; run forward again.', file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        subprocess.run(control + ['-O', 'cancel', '-L', spec, target], stderr=subprocess.DEVNULL)


def remote(args, command, data=None, capture=False):
    """Run `command` in a login shell on Curiosity, so $WORK and modules are set up."""
    script = 'umask 077; ' + command
    result = subprocess.run(ssh_command(args) + ['bash -lc ' + shlex.quote(script)], input=data,
                            stdout=subprocess.PIPE if capture else None)
    if result.returncode:
        raise SystemExit(result.returncode)
    return result.stdout


def release_files():
    manifest = json.loads((CURIOSITY / 'serve/RELEASE_MANIFEST.json').read_text())
    if build(manifest['release'])['files'] != manifest['files']:
        raise SystemExit('curiosity/serve/RELEASE_MANIFEST.json is stale: run python3 curiosity/tools/update_manifest.py')
    names = ['serve/' + name for name in manifest['files']] + ['serve/RELEASE_MANIFEST.json']
    for folder in ('install', 'diagnostics'):
        names += [p.relative_to(CURIOSITY).as_posix() for p in (CURIOSITY / folder).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts]
    return sorted(names)


def release_archive():
    names = release_files()
    digest = hashlib.sha256()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name in names:
            data = (CURIOSITY / name).read_bytes()
            digest.update(name.encode() + b'\0' + hashlib.sha256(data).digest())
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), (0o700 if name.endswith(('.sh', '.sbatch')) else 0o600)
            archive.addfile(info, io.BytesIO(data))
    return digest.hexdigest()[:16], buffer.getvalue()


def sync(args):
    """Upload this release once; later calls find it by its hash. Returns its remote path."""
    release, data = release_archive()
    path = STATE + '/releases/' + release
    remote(args, 'test -f {0}/serve/RELEASE_MANIFEST.json || {{ rm -rf {0}.tmp && mkdir -p {0}.tmp && '
                 'tar -xzf - -C {0}.tmp && mv {0}.tmp {0}; }}'.format(path), data=data)
    return path


def team(args, release, *words, data=None, capture=False):
    words = list(words) + ['--base', args.base]
    return remote(args, 'python3 {}/serve/team.py {}'.format(release, ' '.join(shlex.quote(w) for w in words)),
                  data=data, capture=capture)


def parser():
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    top.add_argument('--user', default=os.environ.get('CURIOSITY_USER'), help='your CINECA login (CURIOSITY_USER)')
    top.add_argument('--host', default=os.environ.get('CURIOSITY_HOST', 'login01-ext.curiosity.cineca.it'),
                     help='login node you use or were assigned, e.g. login02-ext.curiosity.cineca.it (CURIOSITY_HOST)')
    top.add_argument('--base', default=os.environ.get('CURIOSITY_BASE'),
                     help="team folder on Curiosity, expanded there, e.g. '$WORK/team_vllm' (CURIOSITY_BASE)")
    sub = top.add_subparsers(dest='command', metavar='COMMAND')

    def command(name):
        summary, runs = COMMANDS[name]
        return sub.add_parser(name, help=summary, description=summary[0].upper() + summary[1:] + '.',
                              epilog='On Curiosity: ' + runs, formatter_class=argparse.RawDescriptionHelpFormatter)
    p = command('init')
    p.add_argument('--account', help='Slurm project account, if this cluster needs one (ask your mentor or admin)')
    p.add_argument('--partition', help='GPU partition (default boost_usr_prod)')
    p.add_argument('--qos', help='default GPU QoS (default boost_qos_lprod, 2 h)')
    p.add_argument('--cpu-partition', help='CPU partition with internet, for installs and the router (default lrd_all_serial)')
    p.add_argument('--force', action='store_true', help='overwrite an existing inference-site.json')
    p = command('share')
    p.add_argument('logins', nargs='+', metavar='LOGIN', help='CINECA logins of your teammates')
    p.add_argument('--revoke', action='store_true', help='remove their access instead')
    p = command('install')
    p.add_argument('profiles', nargs='*', metavar='PROFILE', help='profiles from curiosity/serve/profiles/, e.g. qwen06b')
    p.add_argument('--link-client', action='store_true', help='also the chisel client for the cluster link')
    p = command('up')
    p.add_argument('profile', help='a profile from curiosity/serve/profiles/, e.g. qwen235b')
    p.add_argument('--qos', help='e.g. boost_qos_dbg: starts sooner, at most 30 minutes')
    p.add_argument('--time', help='time limit, e.g. 00:30:00 (default: the profile\'s)')
    p.add_argument('--reservation', help='Slurm reservation on workshop days')
    command('status')
    p = command('logs')
    p.add_argument('target', help='job id, profile, served model name, or "router"')
    p.add_argument('--lines', type=int, default=60)
    p.add_argument('--follow', '-f', action='store_true', help='keep following the log')
    p = command('down')
    p.add_argument('target', help='profile, served model name, job id, or "router"')
    p = command('router')
    p.add_argument('action', choices=['up', 'down', 'token'], help='token prints your router token')
    p.add_argument('--hours', type=float, default=12, help='up: keep the router alive this long (default 12)')
    p.add_argument('--restart', action='store_true', help='up: replace a running router')
    p.add_argument('--token-file', help='up/token: first install this token (e.g. from the Kubernetes secret)')
    p.add_argument('--link-auth-file', help='up: first install the cluster link user:password from this file')
    p = command('forward')
    p.add_argument('--local-port', type=int, default=18000, help='port on this laptop (default 18000)')
    p = command('diagnose')
    p.add_argument('check', choices=['probe', 'reach', 'nccl'])
    p.add_argument('--target', help='reach: host:port printed by a running probe')
    p.add_argument('--proxy', help='probe/reach: also try this HTTP proxy, e.g. the hackathon proxy')
    p.add_argument('--link-url', help='reach: also check the cluster link server, e.g. https://<host>/curiosity-link')
    p.add_argument('--runtime', help='nccl: installed runtime to test (default vllm-0.31.0)')
    p = command('stage')
    stage_sub = p.add_subparsers(dest='stage_action', required=True)
    r = stage_sub.add_parser('reserve', help='submit the single long 1-node/8-GPU job')
    r.add_argument('--time', help='wall clock, e.g. 10:00:00 (default 11:00:00)')
    r.add_argument('--qos')
    r.add_argument('--reservation')
    r.add_argument('--restart', action='store_true', help='cancel an existing stage first')
    for act in ('load', 'unload'):
        s = stage_sub.add_parser(act, help='{} a model inside the stage'.format(act))
        s.add_argument('profile')
    stage_sub.add_parser('status', help='show loaded models and the reservation')
    stage_sub.add_parser('down', help='cancel the stage reservation')
    f = stage_sub.add_parser('forward', help='forward a laptop port to the stage router')
    f.add_argument('--local-port', type=int, default=18000)
    command('sync')
    return top


def main():
    top = parser()
    args = top.parse_args()
    if not args.command:
        top.print_help()
        return
    if not args.user:
        top.error('set --user or CURIOSITY_USER to your own CINECA login')
    if not args.base and args.command != 'sync':
        top.error("set --base or CURIOSITY_BASE to the team folder, e.g. '$WORK/team_vllm'")

    release = sync(args)
    if args.command == 'sync':
        print('Release on Curiosity: ' + release.replace(STATE, '~/.curiosity-inference'))
    elif args.command == 'init':
        words = ['init'] + (['--account', args.account] if args.account else []) + (['--force'] if args.force else [])
        for flag in ('partition', 'qos', 'cpu_partition'):
            if getattr(args, flag):
                words += ['--' + flag.replace('_', '-'), getattr(args, flag)]
        team(args, release, *words)
    elif args.command == 'share':
        team(args, release, 'share', *args.logins, *(['--revoke'] if args.revoke else []))
    elif args.command == 'install':
        team(args, release, 'install', *args.profiles, *(['--link-client'] if args.link_client else []))
    elif args.command == 'up':
        words = ['up', args.profile]
        for flag in ('qos', 'time', 'reservation'):
            if getattr(args, flag):
                words += ['--' + flag, getattr(args, flag)]
        team(args, release, *words)
    elif args.command == 'status':
        pretty = ['--color'] if sys.stdout.isatty() and not os.environ.get('NO_COLOR') else []
        team(args, release, 'status', *pretty)
    elif args.command == 'logs':
        team(args, release, 'logs', args.target, '--lines', str(args.lines), *(['--follow'] if args.follow else []))
    elif args.command == 'down':
        team(args, release, 'down', args.target)
    elif args.command == 'diagnose':
        words = ['diagnose', args.check]
        for flag in ('target', 'proxy', 'link_url', 'runtime'):
            if getattr(args, flag):
                words += ['--' + flag.replace('_', '-'), getattr(args, flag)]
        team(args, release, *words)
    elif args.command == 'stage' and args.stage_action == 'forward':
        report = json.loads(team(args, release, 'stage', 'status', '--json', capture=True).decode())
        endpoint = report.get('router') or {}
        if not endpoint.get('host') or not endpoint.get('port'):
            raise SystemExit('Stage router is not reporting yet; reserve the stage and wait, then run stage status')
        print('Stage router at http://127.0.0.1:{}/v1 (token: cli.py router token). Ctrl-C to stop.'.format(
            args.local_port), flush=True)
        forward(args, '127.0.0.1:{}:{}:{}'.format(args.local_port, endpoint['host'], endpoint['port']))
    elif args.command == 'stage':
        words = ['stage', args.stage_action]
        if args.stage_action == 'reserve':
            for flag in ('time', 'qos', 'reservation'):
                if getattr(args, flag):
                    words += ['--' + flag, getattr(args, flag)]
            if getattr(args, 'restart', False):
                words += ['--restart']
        elif args.stage_action in ('load', 'unload'):
            words += [args.profile]
        team(args, release, *words)
    elif args.command == 'router':
        if args.token_file:
            team(args, release, 'router', 'token', '--set', data=Path(args.token_file).read_bytes())
        if args.action == 'token':
            if not args.token_file:
                sys.stdout.write(team(args, release, 'router', 'token', capture=True).decode())
        elif args.action == 'down':
            team(args, release, 'router', 'down')
        else:
            if args.link_auth_file:
                team(args, release, 'router', 'link-auth', data=Path(args.link_auth_file).read_bytes())
            team(args, release, 'router', 'up', '--hours', str(args.hours), *(['--restart'] if args.restart else []))
    elif args.command == 'forward':
        report = json.loads(team(args, release, 'status', '--json', capture=True).decode())
        endpoint = (report.get('router') or {}).get('endpoint')
        if not endpoint:
            raise SystemExit('Your router is not listening yet; start it with: router up, then check status')
        print('Router at http://127.0.0.1:{}/v1 (bearer token: cli.py router token). Ctrl-C to stop.'.format(args.local_port), flush=True)
        forward(args, '127.0.0.1:{}:{}:{}'.format(args.local_port, endpoint['host'], endpoint['port']))


if __name__ == '__main__':
    main()
