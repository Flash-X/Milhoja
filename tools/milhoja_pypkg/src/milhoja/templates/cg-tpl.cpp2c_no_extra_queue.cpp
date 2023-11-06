/* _connector:release_extra_queue */
int _param:release (void* packet, const int id) {
    std::cerr << "[_param:release] Packet does not have extra queues." << std::endl;
    return MILHOJA_ERROR_UNABLE_TO_RELEASE_STREAM;
}
