"""Install the pinned PP cache-allocation guard before vLLM imports."""
import os
try:
    from pp_kv_guard import install
    install()
except BaseException as error:
    # Python normally ignores sitecustomize failures. This runtime must fail closed.
    os.write(2, ('PP_KV_GUARD_INSTALL_FAILED: ' + str(error) + '\n').encode())
    os._exit(78)
