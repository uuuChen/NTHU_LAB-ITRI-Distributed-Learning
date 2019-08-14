
COMMON_SOCKET_ARGS = {
    'host': 'localhost',
    'port': 8080,
    'buffer_size': 4096,
    'back_log': 5,
    'max_buffer_size': int(1e10)
}

SERVER_SOCKET_ARGS = {
    'type': 'server'
}

AGENT_SOCKET_ARGS = {
    'type': 'client'
}

SERVER_SOCKET_ARGS.update(COMMON_SOCKET_ARGS)
AGENT_SOCKET_ARGS.update(COMMON_SOCKET_ARGS)



