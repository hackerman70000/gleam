from pathlib import Path

import moderngl
import numpy as np
from loguru import logger
from pyrr import Matrix44

from gleam.config import IMG_SIZE, MODELS_DIR, SHADERS_DIR, CameraConfig
from gleam.renderer.base import Color3, Renderer, Vec3
from gleam.renderer.mesh import Mesh, load_obj


def _load_shader_program(ctx: moderngl.Context, shader_dir: Path, name: str) -> moderngl.Program:
    vert_path = shader_dir / f"{name}.vert"
    frag_path = shader_dir / f"{name}.frag"
    with open(vert_path) as vf, open(frag_path) as ff:
        return ctx.program(vertex_shader=vf.read(), fragment_shader=ff.read())


def _make_context() -> moderngl.Context:
    """Create a headless OpenGL context.

    On Linux (DGX, CI, servers) we first try the EGL backend, which does not
    need a display. macOS ignores ``backend='egl'`` — falling through to the
    default standalone path gives a CGL context there.
    """
    try:
        return moderngl.create_context(standalone=True, backend="egl")
    except Exception as exc:
        logger.debug(f"EGL context unavailable ({exc!r}); falling back to default backend")
        return moderngl.create_standalone_context()


class GLSLRenderer(Renderer):
    """Headless ModernGL renderer that evaluates the Phong shader.

    Usage::

        with GLSLRenderer() as r:
            img = r.render(obj_pos, light_pos, kd_255, shininess)
    """

    def __init__(
        self,
        camera: CameraConfig | None = None,
        image_size: int = IMG_SIZE,
        mesh_path: Path = MODELS_DIR / "sphere.obj",
        shader_dir: Path = SHADERS_DIR / "phong",
        shader_name: str = "phong",
    ) -> None:
        self.camera = camera or CameraConfig()
        self.image_size = image_size
        self.ctx = _make_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        self._color_rb = self.ctx.renderbuffer((image_size, image_size), components=4)
        self._depth_rb = self.ctx.depth_renderbuffer((image_size, image_size))
        self._fbo = self.ctx.framebuffer(
            color_attachments=[self._color_rb], depth_attachment=self._depth_rb
        )

        self._program = _load_shader_program(self.ctx, shader_dir, shader_name)
        self._mesh: Mesh = load_obj(mesh_path)
        self._vbo_pos = self.ctx.buffer(self._mesh.positions.tobytes())
        self._vbo_nor = self.ctx.buffer(self._mesh.normals.tobytes())
        self._vao = self.ctx.vertex_array(
            self._program,
            [
                (self._vbo_pos, "3f", "in_position"),
                (self._vbo_nor, "3f", "in_normal"),
            ],
        )

        proj = Matrix44.perspective_projection(
            self.camera.fovy_deg,
            1.0,
            self.camera.near,
            self.camera.far,
            dtype="f4",
        )
        view = Matrix44.look_at(
            self.camera.eye,
            self.camera.target,
            self.camera.up,
            dtype="f4",
        )
        self._vp: Matrix44 = proj * view
        self._identity3 = np.eye(3, dtype="f4")

    def render(
        self,
        object_pos: Vec3,
        light_pos: Vec3,
        kd_255: Color3,
        shininess: float,
    ) -> np.ndarray:
        model = Matrix44.from_translation(
            np.asarray(object_pos, dtype="f4"), dtype="f4"
        )
        mvp = self._vp * model

        self._program["u_mvp"].write(np.asarray(mvp, dtype="f4").tobytes())
        self._program["u_model"].write(np.asarray(model, dtype="f4").tobytes())
        self._program["u_normal_mat"].write(self._identity3.tobytes())
        self._program["u_kd"].value = tuple(c / 255.0 for c in kd_255)
        self._program["u_n"].value = float(shininess)
        self._program["u_light_pos"].value = tuple(float(v) for v in light_pos)
        self._program["u_cam_pos"].value = tuple(float(v) for v in self.camera.eye)

        self._fbo.use()
        self.ctx.viewport = (0, 0, self.image_size, self.image_size)
        self._fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        self._vao.render(moderngl.TRIANGLES)

        raw = self._fbo.read(components=3, dtype="f1")
        img = np.frombuffer(raw, dtype=np.uint8).reshape(self.image_size, self.image_size, 3)
        return np.ascontiguousarray(np.flipud(img))

    def close(self) -> None:
        for obj in (
            getattr(self, "_vao", None),
            getattr(self, "_vbo_pos", None),
            getattr(self, "_vbo_nor", None),
            getattr(self, "_fbo", None),
            getattr(self, "_color_rb", None),
            getattr(self, "_depth_rb", None),
            getattr(self, "_program", None),
            getattr(self, "ctx", None),
        ):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
