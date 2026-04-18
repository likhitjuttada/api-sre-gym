import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.action_parser import parse, canonical_hash_key


def main() -> None:
    cases = [
        ('GET /users', 'http', 'GET', '/users', None),
        ('POST /login {"user":"admin","password":"admin"}', 'http', 'POST', '/login', {'user': 'admin', 'password': 'admin'}),
        ('curl -X GET http://host/admin/stats', 'http', 'GET', '/admin/stats', None),
        ('curl -X POST http://host/transfer -d \'{"src":"alice","dst":"bob","amount":50}\'', 'http', 'POST', '/transfer', {'src': 'alice', 'dst': 'bob', 'amount': 50}),
        ('FOUND: /admin/stats has no auth', 'found', None, None, None),
        ('```python\nGET /api/search?q=x\n```', 'http', 'GET', '/api/search?q=x', None),
        ('', 'invalid', None, None, None),
        ('just rambling', 'invalid', None, None, None),
    ]
    failures = 0
    for text, want_kind, want_method, want_path, want_body in cases:
        p = parse(text)
        ok = p.kind == want_kind and p.method == want_method and p.path == want_path and p.body == want_body
        status = 'PASS' if ok else 'FAIL'
        if not ok:
            failures += 1
        print(f'{status} kind={p.kind} method={p.method} path={p.path} body={p.body} claim={p.raw_claim} err={p.error}')
    print(f'\n{len(cases) - failures}/{len(cases)} passed')
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
