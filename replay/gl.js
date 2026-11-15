// Small WebGL2 renderer for the swarmsim app: the city as a coloured point cloud,
// vehicles as lit meshes. The world is Z-up, in metres, exactly as in the
// simulator. No external libraries.

export const FOV = 50 * Math.PI / 180;

export const mat4 = {
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  },
  lookAt(e, c, u) {
    let zx = e[0] - c[0], zy = e[1] - c[1], zz = e[2] - c[2];
    let l = Math.hypot(zx, zy, zz) || 1; zx /= l; zy /= l; zz /= l;
    let xx = u[1] * zz - u[2] * zy, xy = u[2] * zx - u[0] * zz, xz = u[0] * zy - u[1] * zx;
    l = Math.hypot(xx, xy, xz) || 1; xx /= l; xy /= l; xz /= l;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
      -(xx * e[0] + xy * e[1] + xz * e[2]), -(yx * e[0] + yy * e[1] + yz * e[2]), -(zx * e[0] + zy * e[1] + zz * e[2]), 1]);
  },
  mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
    return o;
  },
  // position + row-major 3x3 rotation (as the simulator sends it) + uniform scale
  pose(p, R, s = 1) {
    return new Float32Array([R[0] * s, R[3] * s, R[6] * s, 0, R[1] * s, R[4] * s, R[7] * s, 0,
      R[2] * s, R[5] * s, R[8] * s, 0, p[0], p[1], p[2], 1]);
  },
  place(p, s = 1) {
    return new Float32Array([s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, p[0], p[1], p[2], 1]);
  },
};

const VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUV;
uniform mat4 uVP, uModel;
out vec3 vWorld; out vec2 vUV;
void main() {
  vec4 w = uModel * vec4(aPos, 1.0);
  vWorld = w.xyz; vUV = aUV;
  gl_Position = uVP * w;
}`;

// uMode: 0 = textured and lit, 1 = plain colour and lit, 2 = plain colour, unlit
const FS = `#version 300 es
precision highp float;
in vec3 vWorld; in vec2 vUV;
uniform sampler2D uTex;
uniform vec4 uColor;
uniform int uMode;
uniform vec3 uEye, uLight, uFog;
uniform float uFogDensity;
out vec4 frag;
void main() {
  vec4 base = uMode == 0 ? texture(uTex, vUV) * uColor : uColor;
  if (uMode == 0 && base.a < 0.35) discard;
  vec3 c = base.rgb;
  if (uMode != 2) {
    vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
    if (dot(n, uEye - vWorld) < 0.0) n = -n;
    c *= 0.58 + 0.55 * max(dot(n, uLight), 0.0);
  }
  float fog = 1.0 - exp(-length(uEye - vWorld) * uFogDensity);
  frag = vec4(mix(c, uFog, clamp(fog, 0.0, 1.0)), base.a);
}`;

// The city as a point cloud: every point keeps its texture colour and is sized so
// neighbouring points just about touch up close; far away they shrink to a pixel.
const PVS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aCol;
uniform mat4 uVP;
uniform vec3 uEye;
uniform float uPxPerM, uSize, uMinPx, uMaxPx, uFogDensity;
out vec3 vCol; out float vFog;
void main() {
  vec4 c = uVP * vec4(aPos, 1.0);
  gl_Position = c;
  gl_PointSize = clamp(uSize * uPxPerM / max(c.w, 0.1), uMinPx, uMaxPx);
  vCol = aCol.rgb;
  vFog = clamp(1.0 - exp(-length(uEye - aPos) * uFogDensity), 0.0, 1.0);
}`;
const PFS = `#version 300 es
precision mediump float;
in vec3 vCol; in float vFog;
uniform vec3 uFog;
out vec4 frag;
void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  if (dot(q, q) > 1.0) discard;
  frag = vec4(mix(vCol, uFog, vFog), 1.0);
}`;

function program(gl, vs, fs, names) {
  const prog = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    gl.attachShader(prog, sh);
  }
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  const u = {};
  for (const n of names) u[n] = gl.getUniformLocation(prog, n);
  return { prog, u };
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = this.gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
    if (!gl) throw new Error('This browser does not support WebGL2, which the 3-D view needs.');
    const mesh = program(gl, VS, FS, ['uVP', 'uModel', 'uTex', 'uColor', 'uMode', 'uEye', 'uLight', 'uFog', 'uFogDensity']);
    this.prog = mesh.prog; this.u = mesh.u;
    this.points = program(gl, PVS, PFS, ['uVP', 'uEye', 'uPxPerM', 'uSize', 'uMinPx', 'uMaxPx', 'uFog', 'uFogDensity']);
    this.maxPointPx = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1];
    this.sky = [0, 0, 0];
    this.identity = mat4.place([0, 0, 0]);
    this.city = [];
    this.fogDensity = 0.001;
    this.sphere = this.mesh(sphereVertices(14, 20));
    this.lineVao = gl.createVertexArray();
    this.lineBuf = gl.createBuffer();
    gl.bindVertexArray(this.lineVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
    gl.bindVertexArray(null);
  }

  // data: Float32Array of [x, y, z, u, v] per vertex, three vertices per triangle
  mesh(data) {
    const gl = this.gl, vao = gl.createVertexArray(), buf = gl.createBuffer();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 20, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 20, 12);
    gl.bindVertexArray(null);
    return { vao, buf, count: data.length / 5 };
  }

  async fetchMesh(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Could not load ${url}`);
    return this.mesh(new Float32Array(await r.arrayBuffer()));
  }

  // Point buffer: per point 3 floats (x, y, z) and 4 bytes of colour, 16 bytes in all.
  async fetchCloud(c) {
    const r = await fetch(c.url);
    if (!r.ok) throw new Error(`Could not load ${c.url}`);
    const data = await r.arrayBuffer();         // all GL calls after the last await: clouds load in parallel
    const gl = this.gl, vao = gl.createVertexArray(), buf = gl.createBuffer();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.UNSIGNED_BYTE, true, 16, 12);
    gl.bindVertexArray(null);
    return { vao, buf, count: c.count, spacing: c.spacing, kind: c.kind };
  }

  // Loads everything a scene description from /api/scene lists. Replaces the previous scene.
  async loadScene(scene) {
    const gl = this.gl;
    const [city, quad, wing, defence] = await Promise.all([
      Promise.all(scene.clouds.map((c) => this.fetchCloud(c))),
      this.fetchMesh(scene.vehicles.quad), this.fetchMesh(scene.vehicles.wing), this.fetchMesh(scene.vehicles.defence),
    ]);
    for (const old of this.city) { gl.deleteVertexArray(old.vao); gl.deleteBuffer(old.buf); }
    this.city = city;
    this.vehicles = { quad, wing, defence };
    this.size = scene.size;
    this.fogDensity = 0.3 / scene.size;
  }

  resize() {
    const c = this.canvas, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  }

  begin(eye, target) {
    const gl = this.gl, c = this.canvas;
    this.resize();
    gl.viewport(0, 0, c.width, c.height);
    gl.clearColor(...this.sky, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    const dist = Math.hypot(eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]);
    const proj = mat4.perspective(FOV, c.width / c.height, Math.max(0.3, dist * 0.02), Math.max(4000, dist * 20));
    this.vp = mat4.mul(proj, mat4.lookAt(eye, target, [0, 0, 1]));
    this.eye = eye;
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.u.uVP, false, this.vp);
    gl.uniform3fv(this.u.uEye, eye);
    gl.uniform3f(this.u.uLight, 0.36, -0.48, 0.8);
    gl.uniform3fv(this.u.uFog, this.sky);
    gl.uniform1f(this.u.uFogDensity, this.fogDensity);
    gl.uniform1i(this.u.uTex, 0);
    gl.activeTexture(gl.TEXTURE0);
  }

  drawCity() {
    const gl = this.gl, P = this.points, dpr = this.canvas.width / Math.max(1, this.canvas.clientWidth);
    gl.useProgram(P.prog);
    gl.uniformMatrix4fv(P.u.uVP, false, this.vp);
    gl.uniform3fv(P.u.uEye, this.eye);
    gl.uniform3fv(P.u.uFog, this.sky);
    gl.uniform1f(P.u.uFogDensity, this.fogDensity);
    gl.uniform1f(P.u.uPxPerM, this.canvas.height / (2 * Math.tan(FOV / 2)));
    gl.uniform1f(P.u.uMinPx, 1.2 * dpr);
    gl.uniform1f(P.u.uMaxPx, Math.min(7 * dpr, this.maxPointPx));
    for (const c of this.city) {
      gl.uniform1f(P.u.uSize, c.spacing * 0.85);       // a little smaller than the gaps: it reads as points
      gl.bindVertexArray(c.vao);
      gl.drawArrays(gl.POINTS, 0, c.count);
    }
    gl.bindVertexArray(null);
    gl.useProgram(this.prog);
  }

  // mode as in the fragment shader; colours with alpha < 1 are blended and do not write depth
  draw(mesh, model, color, mode = 1, tex = null) {
    const gl = this.gl, blend = color[3] < 1;
    if (blend) { gl.enable(gl.BLEND); gl.depthMask(false); }
    if (tex) gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniformMatrix4fv(this.u.uModel, false, model);
    gl.uniform4fv(this.u.uColor, color);
    gl.uniform1i(this.u.uMode, mode);
    gl.bindVertexArray(mesh.vao);
    gl.drawArrays(gl.TRIANGLES, 0, mesh.count);
    if (blend) { gl.disable(gl.BLEND); gl.depthMask(true); }
  }

  // points: flat [x, y, z, ...]; drawn as one connected line, or as separate segments when strip is false.
  // onTop: drawn over everything (guides that must stay visible through the point cloud).
  lines(points, color, strip = true, onTop = false) {
    const gl = this.gl;
    if (points.length < 6) return;
    if (onTop) gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.uniformMatrix4fv(this.u.uModel, false, this.identity);
    gl.uniform4fv(this.u.uColor, color);
    gl.uniform1i(this.u.uMode, 2);
    gl.bindVertexArray(this.lineVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
    gl.bufferData(gl.ARRAY_BUFFER, points instanceof Float32Array ? points : new Float32Array(points), gl.DYNAMIC_DRAW);
    gl.drawArrays(strip ? gl.LINE_STRIP : gl.LINES, 0, points.length / 3);
    gl.disable(gl.BLEND);
    if (onTop) gl.enable(gl.DEPTH_TEST);
  }

  // World point -> CSS pixel position on the canvas, or null when it is behind the camera.
  project(p) {
    const m = this.vp, x = p[0], y = p[1], z = p[2];
    const w = m[3] * x + m[7] * y + m[11] * z + m[15];
    if (w <= 0.01) return null;
    const cx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w, cy = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
    return [(cx * 0.5 + 0.5) * this.canvas.clientWidth, (0.5 - cy * 0.5) * this.canvas.clientHeight, w];
  }

  // Scale factor that keeps something `size` metres across at least `px` pixels wide on screen.
  minScale(p, size, px, cap = 60) {
    const d = Math.hypot(p[0] - this.eye[0], p[1] - this.eye[1], p[2] - this.eye[2]);
    return Math.min(cap, Math.max(1, px * 2 * Math.tan(FOV / 2) * d / (this.canvas.clientHeight * size)));
  }
}

function sphereVertices(rows, cols) {
  const out = [], pt = (i, j) => {
    const a = Math.PI * i / rows, b = 2 * Math.PI * j / cols;
    return [Math.sin(a) * Math.cos(b), Math.sin(a) * Math.sin(b), Math.cos(a), 0, 0];
  };
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    out.push(...pt(i, j), ...pt(i + 1, j), ...pt(i + 1, j + 1), ...pt(i, j), ...pt(i + 1, j + 1), ...pt(i, j + 1));
  }
  return new Float32Array(out);
}
