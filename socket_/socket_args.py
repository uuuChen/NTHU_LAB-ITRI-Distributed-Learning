
COMMON_SOCKET_ARGS = {
    'buffer_size': 2048,
    'back_log': 5,
}

SERVER_SOCKET_ARGS = {
    'type': 'server',
    # 'host': 'localhost',
    # 'port': 8080,
    'ip': 'localhost/8080',
}

AGENT_1_SOCKET_ARGS = {
    'type': 'client'
}

SERVER_SOCKET_ARGS.update(COMMON_SOCKET_ARGS)
AGENT_1_SOCKET_ARGS.update(COMMON_SOCKET_ARGS)



