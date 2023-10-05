/* _connector:release_extra_queue */
int _param:release (void* packet, const int id) {
    if (packet == nullptr) {
        std::cerr << "[_param:release] packet is NULL" << std::endl;
        return MILHOJA_ERROR_POINTER_IS_NULL;
    }
    _param:class_name*   packet_h = static_cast<_param:class_name*>(packet);

    try {
        _param:release
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return MILHOJA_ERROR_UNABLE_TO_RELEASE_STREAM;
    } catch (...) {
        std::cerr << "[_param:release] Unknown error caught" << std::endl;
        return MILHOJA_ERROR_UNABLE_TO_RELEASE_STREAM;
    }

    return MILHOJA_SUCCESS;
}
