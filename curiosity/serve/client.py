"""Portable client: private key entry, SSH forwarding, English chat and API tests."""
import argparse
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

from inference import ROOT, identifier, load, profile_check


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['key', 'tunnel', 'chat', 'test'])
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, default=Path('.runtime/api_key'))
    parser.add_argument('--endpoint', type=Path)
    parser.add_argument('--ssh-host', default='login01-ext.leonardo.cineca.it')
    parser.add_argument('--user')
    parser.add_argument('--local-port', type=int)
    parser.add_argument('--base-url')
    parser.add_argument('--output', type=Path, default=Path('evidence/client-tests.json'))
    args = parser.parse_args()
    profile = profile_check(load(args.profile))
    port = args.local_port or profile['client']['local_port']
    if not 1 <= port <= 65535:
        raise ValueError('Invalid local port')
    if args.command == 'key':
        args.key_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        key = getpass.getpass('Session inference API key (hidden): ').strip()
        if not key:
            raise ValueError('Empty key')
        fd = os.open(str(args.key_file), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as handle:
            handle.write(key + '\n')
        print('Private key file: ' + str(args.key_file.resolve()))
        return
    if args.command == 'tunnel':
        if not args.endpoint or not args.user:
            raise ValueError('Tunnel requires --endpoint and your own --user')
        endpoint = load(args.endpoint)
        if endpoint['model'] != profile['served_model']:
            raise ValueError('Endpoint model differs from profile')
        remote_port = int(endpoint['port'])
        if not 1 <= remote_port <= 65535:
            raise ValueError('Invalid remote port')
        host, user, node = map(identifier, (args.ssh_host, args.user, endpoint['host']))
        command = ['ssh', '-N', '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=30',
                   '-o', 'ServerAliveCountMax=3', '-L', '127.0.0.1:{}:{}:{}'.format(port, node, remote_port), user + '@' + host]
        print('Keep this terminal open. API: http://127.0.0.1:{}'.format(port), flush=True)
        subprocess.check_call(command)
        return
    base = args.base_url or 'http://127.0.0.1:' + str(port)
    if args.command == 'test':
        subprocess.check_call([sys.executable, str(ROOT / 'test_endpoint.py'), '--profile', str(args.profile.resolve()),
                               '--base-url', base, '--api-key-file', str(args.key_file), '--output', str(args.output)])
        return
    key = args.key_file.read_text().strip()
    body = {'model': profile['served_model'], 'messages': [{'role': 'user', 'content': 'In English, briefly explain what an HPC inference service does.'}],
            'temperature': 0, 'max_tokens': 256, 'chat_template_kwargs': profile['client']['chat_template_kwargs']}
    request = urllib.request.Request(base.rstrip('/') + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=profile['client']['timeout_seconds']) as response:
        print(json.load(response)['choices'][0]['message']['content'])


if __name__ == '__main__':
    main()
