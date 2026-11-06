// Read-only telemetry against the pinned PySuperTuxKart2 headers.
#include "modes/world.hpp"
#include "karts/abstract_kart.hpp"
#include "karts/abstract_kart_animation.hpp"

extern "C" double kartstudy_rescue_time_left(int kart_id) {
    World* world = World::getWorld();
    if (!world || kart_id < 0 || kart_id >= (int)world->getNumKarts()) return -1;
    auto animation = world->getKart(kart_id)->getKartAnimation();
    if (!animation || animation->getAnimationType() != KAT_RESCUE) return -1;
    return animation->getAnimationTimer();
}
