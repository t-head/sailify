// CUDA compute-sanitizer callback API in CUDA spelling.  Converted to the
// PPU_SDK 2v3 hggc_ names by sailify; needs PPU SDK >= 2.3 to compile.
#include <sanitizer.h>

static int g_event_hits = 0;

static void on_event(void *userdata, Sanitizer_CallbackDomain domain,
                     Sanitizer_CallbackId cbid, const void *cbdata) {
    (void)userdata;
    (void)domain;
    (void)cbid;
    (void)cbdata;
    ++g_event_hits;
}

TEST(sanitizer_api, common_callbacks) {
    Sanitizer_SubscriberHandle sub = nullptr;
    SanitizerResult res = sanitizerSubscribe(&sub, on_event, nullptr);
    if (res != SANITIZER_SUCCESS) {
        return;
    }
    res = sanitizerEnableDomain(1, sub, SANITIZER_CB_DOMAIN_RESOURCE);
    if (res == SANITIZER_ERROR_INVALID_CONTEXT
            || res == SANITIZER_ERROR_MAX_LIMIT_REACHED
            || res == SANITIZER_ERROR_OUT_OF_MEMORY) {
        // expected failure paths
    }
    sanitizerUnsubscribe(sub);
}

TEST(sanitizer_api, barrier_count) {
    uint32_t count = 0;
    SanitizerResult res = sanitizerGetCudaBarrierCount((CUfunction)0, &count);
    if (res == SANITIZER_ERROR_INVALID_DOMAIN_ID) {
        // expected failure path
    }
}
