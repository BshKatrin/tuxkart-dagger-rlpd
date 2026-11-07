// Read-only telemetry against the pinned PySuperTuxKart2 headers.
#include "modes/world.hpp"
#include "karts/abstract_kart.hpp"
#include "karts/abstract_kart_animation.hpp"

extern "C" int kartstudy_telemetry_cpp_string_abi() {
#if defined(__GLIBCXX__) && defined(_GLIBCXX_USE_CXX11_ABI)
    return _GLIBCXX_USE_CXX11_ABI;
#else
    return -1;  // macOS uses libc++; Linux must match the wheel's libstdc++ ABI.
#endif
}

extern "C" double kartstudy_rescue_time_left(int kart_id) {
    World* world = World::getWorld();
    if (!world || kart_id < 0 || kart_id >= (int)world->getNumKarts()) return -1;
    auto animation = world->getKart(kart_id)->getKartAnimation();
    if (!animation || animation->getAnimationType() != KAT_RESCUE) return -1;
    return animation->getAnimationTimer();
}
