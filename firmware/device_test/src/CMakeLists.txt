idf_component_register(
    SRCS "main.cpp" "../../lib/zt_logmel/zt_logmel.cpp" "../../lib/zt_calib/zt_calib.cpp"
    INCLUDE_DIRS "." "../../lib/zt_logmel" "../../lib/zt_calib" "../../generated"
    REQUIRES esp_timer
    EMBED_FILES "../../generated/bench_clips.raw")
