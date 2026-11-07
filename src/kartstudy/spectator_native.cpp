// Presentation-only adapter against checksum-verified PySuperTuxKart2 0.7.4 headers.
#include "modes/world.hpp"
#include "states_screens/state_manager.hpp"
#include "karts/abstract_kart.hpp"
#include "karts/kart_model.hpp"
#include "karts/kart_properties.hpp"
#include "karts/kart_properties_manager.hpp"
#include "graphics/irr_driver.hpp"
#include "graphics/lod_node.hpp"
#include "graphics/render_target.hpp"
#include "graphics/shader_based_renderer.hpp"
#include "graphics/frame_buffer.hpp"
#include "ge_render_info.hpp"
#include "ISceneManager.h"
#include "ICameraSceneNode.h"
#include <memory>
#include <stdexcept>
#include <vector>
#include <string>

namespace {
std::string error;
// Irrlicht clears the parent dirty flag after an update. Explicitly refresh
// descendants, including hidden LOD children, when replacing a frozen pose.
void updateTransforms(irr::scene::ISceneNode* node) {
    node->setNeedsUpdateAbsTrans(true);
    node->updateAbsolutePosition();
    const auto& children = node->getChildren();
    for (unsigned i = 0; i < children.size(); ++i) updateTransforms(children[i]);
}
struct Ghost {
    std::unique_ptr<KartModel> model;
    irr::scene::ISceneNode* node = nullptr;
    ~Ghost() { if (node) node->remove(); }
};
struct Presentation {
    std::vector<std::unique_ptr<Ghost>> ghosts;
    irr::scene::ICameraSceneNode* camera = nullptr;
    std::unique_ptr<RenderTarget> target;
    int width, height;
    bool fail_next_render = false;
    ~Presentation() {
        ghosts.clear();
        target.reset();
        if (camera) { camera->remove(); camera->drop(); }
    }
};
// STK teardown owns the renderer's current RTT; restore it even on failure.
struct RenderState {
    ShaderBasedRenderer* renderer;
    RTT* rtts;
    irr::scene::ISceneManager* scene;
    irr::scene::ICameraSceneNode* camera;
    GLint read_fbo, draw_fbo, read_buffer, pack, pbo, viewport[4];
    RenderState() {
        renderer = dynamic_cast<ShaderBasedRenderer*>(irr_driver->getRenderer());
        rtts = renderer->getRTTs();
        scene = irr_driver->getSceneManager();
        camera = scene->getActiveCamera();
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read_fbo);
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &draw_fbo);
        glGetIntegerv(GL_READ_BUFFER, &read_buffer);
        glGetIntegerv(GL_PACK_ALIGNMENT, &pack);
        glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &pbo);
        glGetIntegerv(GL_VIEWPORT, viewport);
    }
    ~RenderState() {
        renderer->setRTT(rtts);
        scene->setActiveCamera(camera);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, draw_fbo);
        glReadBuffer(read_buffer);
        glPixelStorei(GL_PACK_ALIGNMENT, pack);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
        glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    }
};
template<class F> int checked(F f) {
    try { error.clear(); f(); return 0; }
    catch (const std::exception& e) { error = e.what(); return -1; }
}
}
extern "C" {
const char* kartstudy_spectator_error() { return error.c_str(); }
void kartstudy_spectator_release_players() {
    // PySTK cleanup deletes profiles but leaves StateManager active-player pointers.
    // Drop these owned presentation players before deleting their profiles.
    StateManager::get()->resetActivePlayers();
}
int kartstudy_spectator_count() {
    auto w = World::getWorld(); return w ? (int)w->getNumKarts() : -1;
}
double kartstudy_spectator_time() {
    auto w = World::getWorld(); return w ? w->getTime() : -1;
}
int kartstudy_spectator_ticks() {
    auto w = World::getWorld(); return w ? w->getTicksSinceStart() : -1;
}
void* kartstudy_spectator_create(int width, int height) {
    Presentation* result = nullptr;
    checked([&] {
        auto w = World::getWorld();
        if (!w || !irr_driver) throw std::runtime_error("No rendered native world");
        if (!dynamic_cast<ShaderBasedRenderer*>(irr_driver->getRenderer()))
            throw std::runtime_error("Spectator requires the native GL3 renderer");
        auto p = std::make_unique<Presentation>();
        p->width = width; p->height = height;
        p->camera = irr_driver->getSceneManager()->addCameraSceneNode();
        p->camera->grab();
        p->camera->setNearValue(0.1f); p->camera->setFarValue(2000.f);
        p->camera->setAspectRatio(float(width) / height);
        p->target = irr_driver->createRenderTarget(
            irr::core::dimension2du(width, height), "kartstudy-spectator");
        if (!dynamic_cast<GL3RenderTarget*>(p->target.get()))
            throw std::runtime_error("Native offscreen GL3 target unavailable");
        // The single world kart is only the engine's loading anchor, never a ghost.
        for (unsigned i = 0; i < w->getNumKarts(); ++i)
            w->getKart(i)->getNode()->setVisible(false);
        result = p.release();
    });
    return result;
}
int kartstudy_spectator_add(void* handle, const char* kart, float hue) {
    return checked([&] {
        auto& p = *static_cast<Presentation*>(handle);
        auto properties = kart_properties_manager->getKart(kart);
        if (!properties) throw std::runtime_error("Unknown native kart model");
        kart_properties_manager->onDemandLoadKartTextures({kart}, false);
        auto g = std::make_unique<Ghost>();
        g->model.reset(properties->getKartModelCopy(
            std::make_shared<GE::GERenderInfo>(hue, false)));
        g->node = g->model->attachModel(true, false);
        if (!g->node) throw std::runtime_error("Native kart model attachment failed");
        // Kart LOD reads Camera::s_active_camera, not our offscreen camera.
        // Keep the single original model enabled; ordinary frustum/scene occlusion remains.
        if (auto lod = dynamic_cast<LODNode*>(g->node)) lod->forceLevelOfDetail(0);
        g->model->update(0, 0, 0, 0, 0);
        g->node->setVisible(false);
        p.ghosts.push_back(std::move(g));
    });
}
int kartstudy_spectator_camera(void* handle, const float* pos, const float* target, float fov) {
    return checked([&] {
        auto c = static_cast<Presentation*>(handle)->camera;
        c->setPosition({pos[0], pos[1], pos[2]});
        c->setTarget({target[0], target[1], target[2]});
        c->setFOV(fov);
        c->updateAbsolutePosition();
    });
}
int kartstudy_spectator_pose(void* handle, int index, const float* pos,
                            const float* xyzw, float speed, float steer, int visible) {
    return checked([&] {
        auto& g = *static_cast<Presentation*>(handle)->ghosts.at(index);
        btQuaternion q(xyzw[0], xyzw[1], xyzw[2], xyzw[3]);
        // Match Moveable::updateGraphics, including its 90-degree singularity guard.
        if (btFuzzyZero(q.getX()) && btFuzzyZero(q.getY()-0.70710677f) &&
            btFuzzyZero(q.getZ()) && btFuzzyZero(q.getW()-0.70710677f))
            q.setX(0.000001f);
        Vec3 angles; angles.setHPR(q);
        g.node->setPosition({pos[0], pos[1], pos[2]});
        g.node->setRotation(angles.toIrrHPR());
        g.model->update(0, 0, steer, speed, 0);
        g.node->setVisible(visible != 0);
        updateTransforms(g.node);
    });
}
int kartstudy_spectator_camera_state(void* handle, float* out) {
    return checked([&] {
        auto c = static_cast<Presentation*>(handle)->camera;
        auto pos = c->getPosition(); auto target = c->getTarget();
        out[0] = pos.X; out[1] = pos.Y; out[2] = pos.Z;
        out[3] = target.X; out[4] = target.Y; out[5] = target.Z;
        out[6] = c->getFOV();
    });
}
int kartstudy_spectator_ghost_state(void* handle, int index, float* out) {
    return checked([&] {
        auto node = static_cast<Presentation*>(handle)->ghosts.at(index)->node;
        auto pos = node->getPosition();
        irr::core::vector3df forward(0, 0, 1);
        node->getAbsoluteTransformation().rotateVect(forward);
        out[0] = pos.X; out[1] = pos.Y; out[2] = pos.Z;
        out[3] = forward.X; out[4] = forward.Y; out[5] = forward.Z;
    });
}
int kartstudy_spectator_ghost_bounds(void* handle, int index, float* out) {
    return checked([&] {
        auto node = static_cast<Presentation*>(handle)->ghosts.at(index)->node;
        auto box = node->getTransformedBoundingBox();
        auto mesh = node->getChildren().empty() ? node : node->getChildren()[0];
        auto pos = mesh->getAbsolutePosition();
        out[0] = pos.X; out[1] = pos.Y; out[2] = pos.Z;
        out[3] = box.MinEdge.X; out[4] = box.MinEdge.Y; out[5] = box.MinEdge.Z;
        out[6] = box.MaxEdge.X; out[7] = box.MaxEdge.Y; out[8] = box.MaxEdge.Z;
    });
}
void kartstudy_spectator_test_fail_next_render(void* handle) {
    static_cast<Presentation*>(handle)->fail_next_render = true;
}
int kartstudy_spectator_render(void* handle, unsigned char* rgb) {
    return checked([&] {
        auto& p = *static_cast<Presentation*>(handle);
        RenderState restore;
        p.target->renderToTexture(p.camera, 0);
        auto fb = static_cast<GL3RenderTarget*>(p.target.get())->getFrameBuffer();
        if (!fb || fb->getWidth() != (unsigned)p.width || fb->getHeight() != (unsigned)p.height)
            throw std::runtime_error("Native target size mismatch");
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        fb->bind(); glReadBuffer(GL_COLOR_ATTACHMENT0);
        if (p.fail_next_render) {
            p.fail_next_render = false;
            throw std::runtime_error("Injected native render failure after target activation");
        }
        glReadPixels(0, 0, p.width, p.height, GL_RGB, GL_UNSIGNED_BYTE, rgb);
        auto gl_error = glGetError();
        if (gl_error != GL_NO_ERROR) throw std::runtime_error("Native framebuffer read failed");
    });
}
void kartstudy_spectator_destroy(void* handle) { delete static_cast<Presentation*>(handle); }
}
