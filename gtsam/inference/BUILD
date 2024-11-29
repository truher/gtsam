cc_library(
    name = "inference",
    srcs = glob([
        "*.cpp",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//gtsam:global_includes",
        "//gtsam/3rdparty/CCOLAMD:ccolamd",
        "//gtsam/base",
        "@eigen",
    ],
)
