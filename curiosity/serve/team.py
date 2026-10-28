"""Team-folder commands, run on a Curiosity login node; cli.py runs them for you over SSH.

Every command takes --base, the team folder (e.g. '$WORK/team_vllm'; variables are expanded here).

  init      create the team folder and its inference-site.json
  share     give named users access to the team folder (POSIX ACLs), or take it away
  install   download runtimes and model weights for profiles (CPU-queue jobs with internet)
  up        prepare and submit a model server (inference.py prepare + submit)
  status    your servers, router and install jobs, and the team registry
  logs      show the log of a job, a profile's newest run or the router
  down      cancel servers by profile, served name or job id, or the router
  router    start or stop your router, or print its token
  diagnose  submit a probe, reachability or NCCL check job
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import pwd
import stat
import subprocess
import sys
import time

import inference
from inference import ROOT, identifier, load, load_site, save, site_base, user_dir
import registry
import router
import stage

RELEASE = ROOT.parent  # holds serve/, install/ and diagnostics/


def cpu_partition(site):
    # Curiosity has a CPU queue with internet; on curiosity the GPU nodes have it too, so the
    # site partition is the fallback when the site file names no CPU partition.
    return site.get('cpu_partition') or site.get('partition') or 'lrd_all_serial'


def script_job(base, site, script, arguments, name, output, partition, dependency=None, environment=None):
    """Submit a job script from install/ or diagnostics/ with the site's account."""
    account = os.environ.get('SBATCH_ACCOUNT') or site.get('account')
    if account == 'YOUR_PROJECT':
        raise ValueError('Set "account" in {}/inference-site.json'.format(base))
    exports = ['ALL', 'BASE=' + str(base), 'CURIOSITY_RELEASE=' + str(RELEASE)]
    exports += ['{}={}'.format(k, v) for k, v in sorted((environment or {}).items())]
    if any(',' in value for value in exports):
        raise ValueError('Values passed to jobs must not contain commas')
    command = ['sbatch', '--parsable', '--job-name=' + name,
               '--output=' + str(output), '--export=' + ','.join(exports), '--partition=' + identifier(partition)]
    if account:
        command.append('--account=' + identifier(account))
    if dependency:
        command.append('--dependency=afterok:' + dependency)
    marker = inference.private_dir(user_dir(base)) / ('.submitting-' + name)
    job = inference.sbatch(command + [str(script)] + list(arguments), marker)
    print('Submitted {} as job {}; log: {}'.format(name, job, str(output).replace('%x-%j', name + '-' + job)))
    return job


def init(args):
    base = site_base(args.base)
    site_file = base / 'inference-site.json'
    if site_file.exists() and not args.force:
        raise ValueError('{} exists already; edit it, or use --force to overwrite it'.format(site_file))
    site = load(ROOT / 'site.example.json')
    if args.account:
        site['account'] = identifier(args.account)
    for key in ('partition', 'qos', 'cpu_partition'):
        if getattr(args, key):
            site[key] = identifier(getattr(args, key))
    base.mkdir(parents=True, exist_ok=True)
    for name in ('runtimes', 'checkpoints', 'endpoints', 'users', 'logs', 'tools'):
        (base / name).mkdir(exist_ok=True)
    # The team root and everything but the per-member trees must be group usable even
    # when a first member created it under umask 077 (0700 would lock teammates out of
    # sbatch logs, checkpoints and runtimes); setgid keeps inheritance on the team group.
    stage.group_open(base, 0o2775)
    for name in ('runtimes', 'checkpoints', 'endpoints', 'logs', 'tools'):
        stage.group_open(base / name, 0o2775)
    save(site_file, site)
    inference.private_dir(user_dir(base))
    print('Team folder ready: {} (account {}).'.format(base, site['account']))
    print('Only you can use it so far; give teammates access with: cli.py share <login> ...')


def setfacl(arguments, targets):
    for start in range(0, len(targets), 500):
        chunk = targets[start:start + 500]
        if subprocess.call(['setfacl'] + arguments + chunk, stderr=subprocess.DEVNULL) == 0:
            continue
        # Files of a running job can disappear meanwhile; retry with what is still there, now with errors shown.
        remaining = [path for path in chunk if os.path.lexists(path)]
        if remaining:
            subprocess.check_call(['setfacl'] + arguments + remaining)


def share(args):
    """Grant (or revoke) named users rwX on the team folder, except other users' personal folders."""
    base = site_base(args.base)
    load_site(base)
    for name in args.users:
        try:
            pwd.getpwnam(name)
        except KeyError:
            raise ValueError('No such user on Curiosity: ' + name)
    files, directories, others = [], [], 0
    for root, dirnames, filenames in os.walk(str(base)):
        here = Path(root)
        if here != base and here.name.startswith('romain'):
            dirnames[:], filenames = [], []  # personal romain* folders stay private to their owner
            continue
        if here == base / 'staging':
            dirnames[:], filenames = [], []  # temporary install areas; new ones inherit from staging/ itself
        if here == base / 'users':
            dirnames[:] = []  # users/<login> stays private to its owner
        for path in [root] + [os.path.join(root, f) for f in filenames]:
            # Only the owner may change a file's ACL, and a symlink's ACL is its target's.
            mode = os.lstat(path)
            if stat.S_ISLNK(mode.st_mode):
                continue
            if mode.st_uid != os.getuid():
                others += 1
                continue
            (directories if path == root else files).append(path)
    inherited = [d for d in directories if Path(d) != base / 'users']
    if args.revoke:
        entries = ','.join('u:' + name for name in args.users)
        setfacl(['-x', entries], files + directories)
        setfacl(['-d', '-x', entries], inherited)
        print('Removed access for: ' + ', '.join(args.users))
        return
    entries = ','.join('u:{}:rwX'.format(name) for name in args.users)
    setfacl(['-m', entries], files + directories)
    # Default entries make new files and folders inherit the access.
    setfacl(['-d', '-m', entries], inherited)
    # Users must be able to pass through the folders above $BASE; add that where we own them.
    for parent in base.parents:
        if parent.stat().st_uid != os.getuid():
            print('Note: {} belongs to someone else; the users also need to be able to enter it.'.format(parent))
            break
        setfacl(['-m', ','.join('u:{}:--x'.format(name) for name in args.users)], [str(parent)])
    if others:
        print('Skipped {} files and folders owned by others; they already inherit the access.'.format(others))
    print('Shared {} with: {}'.format(base, ', '.join(args.users)))


def install(args):
    base = site_base(args.base)
    site = load_site(base)
    jobs = router.active_jobs() or {}
    running = {j['name']: j['job_id'] for j in jobs.values()}
    logs = base / 'logs' / '%x-%j.out'
    runtime_jobs = {}
    if not args.profiles and not args.link_client:
        raise ValueError('Name profiles to install (e.g. gemma4), and/or --link-client')
    for name in args.profiles:
        profile = inference.profile_check(load(ROOT / 'profiles' / (identifier(name) + '.json')))
        runtime = profile['runtime']
        checkpoint = profile['checkpoint']
        if inference.runtime_kind(profile) == 'docker':
            # Docker runtimes have nothing to install locally: the image is pulled on the GPU
            # node at serve time. The checkpoint download runs inside that image (curiosity's
            # GPU nodes have internet), as one job on the site partition.
            if (inference.checkpoint_dir(base, site, checkpoint) / 'download_manifest.json').exists():
                print('checkpoint {}: installed'.format(checkpoint))
            elif 'install-checkpoint-' + checkpoint in running:
                print('checkpoint {}: job {} is already running'.format(checkpoint, running['install-checkpoint-' + checkpoint]))
            else:
                script_job(base, site, RELEASE / 'install/docker_checkpoint.sbatch', [name],
                           'install-checkpoint-' + checkpoint, logs, cpu_partition(site))
            continue
        if (inference.runtime_dir(base, site, runtime) / 'runtime.json').exists():
            print('runtime {}: installed'.format(runtime))
        elif runtime not in runtime_jobs:
            job_name = 'install-runtime-' + runtime
            runtime_jobs[runtime] = running.get(job_name) or script_job(
                base, site, RELEASE / 'install/runtime.sbatch', [runtime], job_name, logs, cpu_partition(site))
        checkpoint = profile['checkpoint']
        if (inference.checkpoint_dir(base, site, checkpoint) / 'download_manifest.json').exists():
            print('checkpoint {}: installed'.format(checkpoint))
        elif 'install-checkpoint-' + checkpoint in running:
            print('checkpoint {}: job {} is already running'.format(checkpoint, running['install-checkpoint-' + checkpoint]))
        else:
            # The download runs inside the profile's own image, so it waits for that runtime.
            script_job(base, site, RELEASE / 'install/checkpoint.sbatch', [name], 'install-checkpoint-' + checkpoint,
                       logs, cpu_partition(site), dependency=runtime_jobs.get(runtime))
    if args.link_client:
        tool = inference.catalog()['tools']['chisel']
        if (base / 'tools' / ('chisel-' + tool['version'])).exists():
            print('cluster link client: installed')
        else:
            script_job(base, site, RELEASE / 'install/link_client.sbatch', [], 'install-link-client', logs, cpu_partition(site))
    print('Follow progress with: cli.py status, and cli.py logs <job id>.')


def up(args):
    run = inference.prepare_run(args.profile, args.base)
    print('Run directory: ' + str(run), flush=True)
    command = [sys.executable, str(run / 'inference.py'), 'submit']
    for flag in ('reservation', 'qos', 'time'):
        if getattr(args, flag):
            command += ['--' + flag, getattr(args, flag)]
    subprocess.check_call(command, cwd=str(run))


_ANSI_STRIP = __import__('re').compile(r'\x1b\[[0-9;]*m')


def _display_width(text):
    """Visible width of text with ANSI escape sequences removed."""
    return len(_ANSI_STRIP.sub('', text))


def render_table(headers, rows, color):
    """A dependency-free grid table (ANSI-safe). Box-drawing glyphs when color is
    on, plain ASCII otherwise so piped output stays safe and greppable."""
    data = [headers] + [[str(cell) for cell in row] for row in rows]
    widths = [max(_display_width(data[r][c]) for r in range(len(data))) for c in range(len(headers))]
    if color:
        tl, tm, tr, ml, mm, mr, bl, bm, br, h, v = '┌┬┐├┼┤└┴┘─│'
    else:
        tl, tm, tr, ml, mm, mr, bl, bm, br, h, v = '+++++++++-|'
    def pad(cell, index):
        gap = widths[index] - _display_width(cell)
        return ' ' + cell + ' ' * (gap + 1) if gap >= 0 else ' ' + cell
    def edge(left, joint, right):
        return left + joint.join(h * (w + 2) for w in widths) + right
    output = [edge(tl, tm, tr)]
    for r, row in enumerate(data):
        output.append(v + v.join(pad(row[c], c) for c in range(len(headers))) + v)
        if r == 0:
            output.append(edge(ml, mm, mr))
    output.append(edge(bl, bm, br))
    return '\n'.join(output)


def _status_art(color, title, subtitle):
    """Colored banner with a small rocket, or a plain header when not colorized."""
    if not color:
        return '\n{0}\n{1}'.format(title, subtitle)
    rocket = ("             \x1b[1;36m▲\x1b[0m\n"
              "            \x1b[1;36m/|\\\x1b[0m\n"
              "           \x1b[1;36m/_|_\\\x1b[0m\n"
              "          \x1b[1;36m▄▄██▄▄\x1b[0m\n"
              "        \x1b[1;36m▄\x1b[0m\x1b[1;38;5;208m████████\x1b[0m\x1b[1;36m▄\x1b[0m\n"
              "      \x1b[1;36m▄▄▀\x1b[0m\x1b[1;38;5;208m▀▀▀▀▀▀▀▀\x1b[0m\x1b[1;36m▀▀▄▄\x1b[0m\n"
              "      \x1b[1;36m█\x1b[0m\x1b[1;38;5;208m █ █████ █ \x1b[0m\x1b[1;36m█\x1b[0m\n"
              "       \x1b[1;36m▀▄▄▄▄▄▄▄▄▄▀\x1b[0m\n"
              "            \x1b[1;36m██\x1b[0m\n"
              "\x1b[1;96mCURIOSITY · INFERENCE STATUS\x1b[0m\n"
              "\x1b[2;90m{}\x1b[0m\n".format(subtitle))
    return rocket


def status(args):
    base = site_base(args.base)
    if not (base / 'inference-site.json').exists():
        if args.json:
            print(json.dumps({'initialised': False}))
        else:
            print('{} is not a team folder yet. Create it with: cli.py init --account <project>'.format(base))
        return
    site = load_site(base)
    jobs = router.active_jobs()
    report = {'initialised': True, 'jobs_known': jobs is not None, 'servers': [], 'router': router.state(base, jobs),
              'installs': [j for j in (jobs or {}).values() if j['name'].startswith('install-')],
              'installed': {'runtimes': sorted(p.parent.name for p in (base / 'runtimes').glob('*/runtime.json')),
                            'checkpoints': sorted(p.parent.name for p in (base / 'checkpoints').glob('*/download_manifest.json'))}}
    for run in router.own_runs(base):
        if run['job'] and (jobs is None or run['job'] in jobs):
            job = (jobs or {}).get(run['job'], {})
            report['servers'].append({'profile': run['profile']['id'], 'model': run['profile']['served_model'],
                                      'job_id': run['job'], 'state': job.get('state', 'unknown'),
                                      'elapsed': job.get('elapsed', '?'), 'limit': job.get('limit', '?'),
                                      'ready': run['endpoint'] is not None, 'run': str(run['run']),
                                      'endpoint': run['endpoint'] and '{}:{}'.format(run['endpoint']['host'], run['endpoint']['port'])})
    shared = registry.directory({'base': str(base), 'site': site})
    report['registry'] = registry.records(shared) if shared else None
    if args.json:
        print(json.dumps(report, indent=2))
        return
    color = args.color
    if jobs is None:
        print(('\x1b[1;33mWARNING:\x1b[0m squeue failed' if color else 'WARNING') + '; showing runs that may already have ended.')

    # Banner
    title = 'Curiosity Inference · status'
    subtitle = '{0} @ {1}'.format(getpass.getuser(), base)
    print(_status_art(color, title, subtitle))

    # Installed assets, one line each: runtimes then checkpoints
    installed = report['installed']
    print(('\x1b[1m' if color else '') + 'Installed' + ('\x1b[0m' if color else ''))
    print('  runtimes ' + (', '.join(installed['runtimes']) if installed['runtimes'] else 'none'))
    print('  checkpoints ' + (', '.join(installed['checkpoints']) if installed['checkpoints'] else 'none'))

    installs = report['installs']
    if installs:
        print()
        for j in installs:
            state_c = {'RUNNING': '1;32', 'PENDING': '1;33', 'FAILED': '1;31'}.get(j['state'], '')
            state = ('\x1b[{0}m{1}\x1b[0m'.format(state_c, j['state']) if state_c and color else j['state'])
            print('  {2} installing {0} [job {1}] {3} {4}'.format(
                j['name'], j['job_id'], '↓' if color else '-', state, j.get('elapsed', '')))

    # Model servers as a table
    print()
    print(('\x1b[1m' if color else '') + 'Your model servers' + ('\x1b[0m' if color else ''))
    headers = ['profile', 'job', 'state', 'time', 'status']
    rows = []
    for s in report['servers']:
        if s['ready']:
            status = ('\x1b[1;32mREADY \x1b[0m' + s['model'] + ' at ' + s['endpoint']) if color else ('READY ' + s['model'] + ' at ' + s['endpoint'])
        else:
            status = ('\x1b[1;33mstarting (not ready yet)\x1b[0m' if color else 'starting (not ready yet)')
        state_colors = {'RUNNING': '1;32', 'PENDING': '1;33', 'COMPLETING': '1;35', 'COMPLETED': '2', 'FAILED': '1;31'}
        state_c = state_colors.get(s['state'], '')
        state = ('\x1b[{0}m{1}\x1b[0m'.format(state_c, s['state']) if state_c and color else s['state'])
        rows.append([s['profile'], s['job_id'], state, '{}/{}'.format(s['elapsed'], s['limit']), status])
    if rows:
        print(render_table(headers, rows, color))
    else:
        print('  none')

    # Router
    print()
    rvar = ('\x1b[1m' if color else '') + 'Your router' + ('\x1b[0m' if color else '')
    if report['router']:
        summary = report['router'].get('summary', '')
        print(rvar + ': ' + summary)
    else:
        print(rvar + ': not running')

    # Team registry
    if report['registry'] is not None:
        print()
        print(('\x1b[1m' if color else '') + 'Team registry (ready servers of everyone sharing this folder)' + ('\x1b[0m' if color else ''))
        if report['registry']:
            rows = []
            for r in report['registry']:
                r = {k: r.get(k, '?') for k in ('name', 'owner', 'host', 'port', 'job_id', 'slurm_expected_end')}
                rows.append([r['name'], r['owner'], '{}:{}'.format(r['host'], r['port']), r['job_id'], r['slurm_expected_end']])
            print(render_table(['name', 'owner', 'endpoint', 'job', 'until'], rows, color))
        else:
            print('  none')


def find_log(base, target):
    mine = user_dir(base)
    if target == 'router':
        candidates = list((mine / 'router').glob('*.out'))
    elif target.isdigit():
        patterns = [(mine / 'runs', '*/*-{}.out'), (mine / 'router', '*-{}.out'), (mine / 'logs', '*-{}.out'),
                    (base / 'logs', '*-{}.out')]
        candidates = [p for root, pattern in patterns for p in root.glob(pattern.format(target))]
    else:
        runs = [r for r in router.own_runs(base) if target in (r['profile']['id'], r['profile']['served_model'])]
        candidates = list(runs[0]['run'].glob('*.out')) if runs else []
    if not candidates:
        raise ValueError('No log found for {} (yet). A pending job has no log until it starts.'.format(target))
    return max(candidates, key=lambda p: p.stat().st_mtime)


def logs(args):
    path = find_log(site_base(args.base), args.target)
    print('==> {} <=='.format(path), flush=True)
    command = ['tail', '-n', str(args.lines)] + (['-f'] if args.follow else []) + [str(path)]
    os.execvp('tail', command)


def down(args):
    base = site_base(args.base)
    if args.target == 'router':
        router.down()
        return
    jobs = router.active_jobs()
    targets = []
    for run in router.own_runs(base):
        p = run['profile']
        if run['job'] and args.target in (p['id'], p['served_model'], run['job']):
            if jobs is None or run['job'] in jobs:
                targets.append(run['job'])
    if not targets:
        raise ValueError('No running server of yours matches {}; see status'.format(args.target))
    inference.ensure_slurm()
    subprocess.check_call(['scancel'] + targets)
    print('Cancelled job(s) ' + ' '.join(targets))


def router_command(args):
    base = site_base(args.base)
    if args.action == 'up':
        router.up(base, load_site(base), args.hours, args.restart, router.active_jobs())
    elif args.action == 'down':
        router.down()
    elif args.action == 'token':
        router.token(base, 'set' if args.set else 'ensure')
        if not args.set:
            router.token(base, 'show')
    elif args.action == 'link-auth':
        router.set_link_auth(base)


def diagnose(args):
    base = site_base(args.base)
    site = load_site(base)
    output = inference.private_dir(inference.private_dir(user_dir(base)) / 'logs') / '%x-%j.out'
    script = RELEASE / 'diagnostics' / {'probe': 'probe.sbatch', 'reach': 'reach.sbatch', 'nccl': 'nccl_check.sbatch'}[args.check]
    environment = {}
    if args.check == 'reach':
        environment = {k: v for k, v in (('TARGET', args.target), ('PROBE_PROXY', args.proxy), ('LINK_URL', args.link_url)) if v}
    elif args.check == 'probe' and args.proxy:
        environment = {'PROBE_PROXY': args.proxy}
    elif args.check == 'nccl':
        environment = {'RUNTIME': args.runtime}
    name = {'probe': 'probe', 'reach': 'reach', 'nccl': 'nccl-check'}[args.check]
    # reach tests what the router's CPU queue can see; probe and NCCL need GPUs (debug QoS, set in the scripts).
    partition = cpu_partition(site) if args.check == 'reach' else site['partition']
    script_job(base, site, script, [], name, output, partition, environment=environment)


def stage_job(base, status):
    """The team's running stage job, seen through the status owner's Slurm view.

    `squeue -u $USER` only shows our own jobs, so a teammate cannot see the stage a
    colleague reserved -- which is exactly why `stage status` returned nothing and,
    worse, `stage reserve` would have submitted a second, colliding stage. The shared
    status.json records the owner; Slurm is then asked from their account (cancelling
    still requires the owner account, as Slurm enforces). Falls back to our own jobs
    for a stage without an owner on record."""
    owner = (status or {}).get('owner') if status else None
    inference.ensure_slurm()
    if owner:
        try:
            output = subprocess.check_output(['squeue', '-h', '-u', owner, '-o', '%i|%j|%T|%M|%l|%N'],
                                             stderr=subprocess.DEVNULL, timeout=30).decode()
        except (OSError, subprocess.SubprocessError):
            output = ''
        for line in output.splitlines():
            fields = line.strip().split('|')
            if len(fields) == 6 and fields[1] == 'stage':
                return dict(zip(('job_id', 'name', 'state', 'elapsed', 'limit', 'nodes'), fields))
    for job in (router.active_jobs() or {}).values():
        if job['name'] == 'stage':
            return job
    return None


def stage_reserve(base, site, time_override, qos, reservation, restart=False):
    """Reattach to the running stage, or submit ONE long 1-node/8-GPU job (never a second)."""
    # The stage is team-shared: any member's status counts, not only our own jobs.
    status = stage.load_status(base)
    job = stage_job(base, status)
    existing = [job] if job else []
    if existing and not restart:
        print('A stage is already reserved as job {} ({}){}; reattaching to it. '
              'Use --restart to replace it instead.'.format(
                  existing[0]['job_id'], existing[0].get('state'),
                  ' by ' + status['owner'] if status and status.get('owner') else ''))
        return
    if restart and existing:
        owner = status.get('owner') if status else None
        if owner and owner != getpass.getuser():
            raise ValueError('The stage reservation belongs to {}; only {} can replace it.'.format(owner, owner))
        subprocess.check_call(['scancel', '--name=stage', '--user=' + (owner or getpass.getuser())])
        cleared = stage.clear_commands(base)
        models_dropped = stage.clear_models(base)
        bits = []
        if cleared:
            bits.append('Cleared {} pending command(s).'.format(cleared))
        if models_dropped:
            bits.append('Forgot {} model record(s) (containers died with the node).'.format(models_dropped))
        print('Replacing existing stage job(s): {}.{}'.format(
            ', '.join(e['job_id'] for e in existing), ' ' + ' '.join(bits) if bits else ''))
    account = os.environ.get('SBATCH_ACCOUNT') or site.get('account')
    logs = stage.stage_dir(base) / 'reserve-%x-%j.out'
    # ensure_dirs, not bare mkdir: the stage dir must already carry the team modes and
    # the setgid bit before slurmd (under any member's account) writes the job log in it.
    stage.ensure_dirs(base)
    command = ['sbatch', '--parsable', '--job-name=stage', '--partition=' + inference.identifier(site['partition']),
               '--nodes=1', '--ntasks-per-node=1', '--gres=gpu:8', '--cpus-per-task=32', '--mem=480GB',
               '--time=' + inference.time_limit(time_override or '11:00:00'),
               '--signal=B:TERM@60', '--output=' + str(logs),
               '--export=' + ','.join(['ALL', 'BASE=' + str(base), 'CURIOSITY_RELEASE=' + str(RELEASE)])]
    if account:
        command.append('--account=' + inference.identifier(account))
    if qos:
        command.append('--qos=' + inference.identifier(qos))
    if reservation:
        command.append('--reservation=' + inference.identifier(reservation))
    job = inference.sbatch(command + [str(RELEASE / 'serve/stage_model.sbatch')],
                           stage.stage_dir(base) / '.submitting-stage')
    print('Reserved stage as job {}: 1 node / 8 B300 for {}.'.format(job, time_override or '11:00:00'))
    print('Inside it: stage load <profile>, stage unload <profile>, stage status.')


def stage_request(base, action, profile):
    d = stage.ensure_dirs(base)
    name = '{}-{}-{}.cmd'.format(int(time.time() * 100000) % 100000, action, inference.identifier(profile))
    path = d / 'cmd' / name
    path.write_text(json.dumps({'action': action, 'profile': profile}) + '\n')
    # The CLI runs with umask 077: without this the supervisor (another member's
    # account, per the shared stage) could not read the command back.
    stage.shared_visible(path)
    print('Queued {} {} -> {} (the supervisor inside the stage acts on it within seconds)'.format(
        action, profile, path))


def stage_status(base, as_json=False):
    d = stage.stage_dir(base)
    models = stage.load_models(base) if d.exists() else {}
    status = stage.load_status(base)
    if as_json:
        cmd_dir = d / 'cmd' if d.exists() and (d / 'cmd').is_dir() else None
        pending = sorted(cmd_dir.glob('*.cmd')) if cmd_dir else []
        print(json.dumps({'supervisor': (status or {}).get('supervisor'),
                          'owner': (status or {}).get('owner'),
                          'router': {'host': (status or {}).get('node'),
                                     'port': status.get('router_port')} if status and status.get('router_port') else None,
                          'models': {k: v['served_model'] for k, v in models.items()},
                          'loading': (status or {}).get('loading'),
                          'pending': [p.name for p in pending]}))
        return
    if status:
        owner = ' by ' + status['owner'] if status.get('owner') else ''
        if status.get('supervisor') == 'down':
            stopped = time.strftime('%H:%M', time.localtime(status.get('updated', time.time())))
            print('supervisor: down since {} (stage down - reserve again to restart){}'.format(stopped, owner))
        else:
            print('supervisor: {} on {} ; router 0.0.0.0:{}{}'.format(
                status.get('supervisor'), status.get('node'), status.get('router_port'), owner))
            loading = status.get('loading')
            if loading:
                age = ''
                started = loading.get('started')
                if started:
                    age = ' ({} min)'.format(int((time.time() - started) // 60))
                detail = ' - ' + loading['detail'] if loading.get('detail') else ''
                print('  loading {:14} slice={:>7} api_port={} phase={}{}{}'.format(
                    loading.get('profile', '?'), loading.get('slice'), loading.get('api_port'),
                    loading.get('phase'), age, detail))
    else:
        print('supervisor: not reported yet (reserve the stage, then wait for its job log)')
    if models:
        for profile, rec in sorted(models.items()):
            print('  {:14} slice={:>7} api_port={} model={}'.format(
                profile, rec['slice'], rec['api_port'], rec['served_model']))
    else:
        print('  no model loaded (stage load <profile>)')
    if d.exists() and (d / 'cmd').is_dir():
        pending = sorted((d / 'cmd').glob('*.cmd'))
        if pending:
            oldest = time.strftime('%H:%M', time.localtime(pending[0].stat().st_mtime))
            print('  pending commands: {} (oldest {}) - supervisor busy or stuck'.format(len(pending), oldest))
    # The stage may have been reserved by a teammate: ask Slurm from the owner's view.
    job = stage_job(base, status)
    if job:
        print('reservation: job {} {}'.format(job['job_id'], job['state']))


def stage_command(args):
    base = site_base(args.base)
    action = args.stage_action
    if action == 'reserve':
        stage_reserve(base, load_site(base), args.time, args.qos, args.reservation,
                      bool(getattr(args, 'restart', False)))
    elif action in ('load', 'unload'):
        stage_request(base, action, args.profile)
    elif action == 'status':
        stage_status(base, bool(getattr(args, 'json', False)))
    elif action == 'down':
        inference.ensure_slurm()
        status = stage.load_status(base)
        owner = status.get('owner') if status else None
        if owner and owner != getpass.getuser():
            raise ValueError('The stage belongs to {}; only {} can down it '
                             '(Slurm grants no cross-account cancellation).'.format(owner, owner))
        subprocess.check_call(['scancel', '--name=stage', '--user=' + (owner or getpass.getuser())])
        cleared = stage.clear_commands(base)
        models_dropped = stage.clear_models(base)
        status_path = stage.stage_dir(base) / 'status.json'
        inference.save(status_path,
                       {'updated': time.time(), 'supervisor': 'down',
                        'router_port': None, 'node': None, 'models': {}, 'loading': None,
                        'owner': owner or getpass.getuser()})
        stage.shared_visible(status_path)
        bits = []
        if cleared:
            bits.append('Cleared {} pending command(s).'.format(cleared))
        if models_dropped:
            bits.append('Forgot {} model record(s) (containers died with the node).'.format(models_dropped))
        print('Cancelled the stage reservation of {}.{}'.format(
            owner or getpass.getuser(), ' ' + ' '.join(bits) if bits else ''))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command')

    def command(name, help_text):
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument('--base', required=True, help="team folder, e.g. '$WORK/team_vllm'")
        return p
    p = command('init', 'create the team folder and inference-site.json')
    p.add_argument('--account', help='Slurm project account, if this cluster needs one')
    p.add_argument('--partition', help='GPU partition (default boost_usr_prod)')
    p.add_argument('--qos', help='default GPU QoS (default boost_qos_lprod)')
    p.add_argument('--cpu-partition', help='CPU partition with internet (default lrd_all_serial)')
    p.add_argument('--force', action='store_true', help='overwrite an existing inference-site.json')
    p = command('share', 'give users access to the team folder (or --revoke it)')
    p.add_argument('users', nargs='+', metavar='LOGIN')
    p.add_argument('--revoke', action='store_true')
    p = command('install', 'download runtimes and weights for profiles')
    p.add_argument('profiles', nargs='*', metavar='PROFILE')
    p.add_argument('--link-client', action='store_true', help='also the chisel client for the cluster link')
    p = command('up', 'prepare and submit a model server')
    p.add_argument('profile')
    p.add_argument('--qos')
    p.add_argument('--time')
    p.add_argument('--reservation')
    p = command('status', 'show servers, router, installs and the team registry')
    p.add_argument('--json', action='store_true')
    p.add_argument('--color', action='store_true', help='colored banner, icons and box-drawing tables')
    p = command('logs', 'show a log: job id, profile, served name or "router"')
    p.add_argument('target')
    p.add_argument('--lines', type=int, default=60)
    p.add_argument('--follow', action='store_true')
    p = command('down', 'cancel servers by profile, served name or job id, or "router"')
    p.add_argument('target')
    p = command('router', 'start/stop the router, print its token, or store the cluster link password')
    p.add_argument('action', choices=['up', 'down', 'token', 'link-auth'])
    p.add_argument('--hours', type=float, default=12)
    p.add_argument('--restart', action='store_true')
    p.add_argument('--set', action='store_true', help='token: replace it with the one on stdin')
    p = command('diagnose', 'submit a probe, reachability or NCCL check job')
    p.add_argument('check', choices=['probe', 'reach', 'nccl'])
    p.add_argument('--target', help='reach: host:port of a running probe')
    p.add_argument('--proxy', help='probe/reach: also try this HTTP proxy')
    p.add_argument('--link-url', help='reach: also check this cluster link URL')
    p.add_argument('--runtime', default='vllm-0.31.0', help='nccl: runtime image to use')
    p = command('stage', 'one long 8-GPU B300 reservation: load/unload models + the router inside')
    p.add_argument('stage_action', choices=['reserve', 'load', 'unload', 'status', 'down'])
    p.add_argument('profile', nargs='?', help='load/unload: the profile to load or stop')
    p.add_argument('--json', action='store_true', help='status: JSON output (used by cli.py stage forward)')
    p.add_argument('--time', help='reserve: wall clock, e.g. 10:00:00 (default 11:00:00)')
    p.add_argument('--qos', help='reserve: slurm QoS')
    p.add_argument('--reservation', help='reserve: slurm reservation')
    p.add_argument('--restart', action='store_true', help='reserve: cancel an existing stage first')
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    {'init': init, 'share': share, 'install': install, 'up': up, 'status': status, 'logs': logs, 'down': down,
     'router': router_command, 'diagnose': diagnose, 'stage': stage_command}[args.command](args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
