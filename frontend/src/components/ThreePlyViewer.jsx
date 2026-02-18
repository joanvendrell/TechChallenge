import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

export default function ThreePlyViewer({ url }) {
  const mountRef = useRef(null);

  useEffect(() => {
    if (!url || !mountRef.current) return;

    const mount = mountRef.current;
    mount.innerHTML = "";

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a1020);

    const camera = new THREE.PerspectiveCamera(50, mount.clientWidth / mount.clientHeight, 0.000001, 1000);
    camera.position.set(1.5, 1.5, 1.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(2, 3, 4);
    scene.add(dir);

    // optional axes
    const axes = new THREE.AxesHelper(1);
    scene.add(axes);

    const loader = new PLYLoader();
    let mesh = null;

    loader.load(
      url,
      (geom) => {
        geom.computeVertexNormals();

        const mat = new THREE.MeshStandardMaterial({ color: 0xffd54a, metalness: 0.1, roughness: 0.7 });
        mesh = new THREE.Mesh(geom, mat);

        // center + scale to a nice size
        geom.computeBoundingBox();
        const bb = geom.boundingBox;
        const center = new THREE.Vector3();
        bb.getCenter(center);
        mesh.position.sub(center);

        const size = new THREE.Vector3();
        bb.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = maxDim > 0 ? 1.0 / maxDim : 1.0;
        mesh.scale.setScalar(scale);

        scene.add(mesh);
      },
      undefined,
      (err) => console.error("PLY load failed:", err)
    );

    let anim = true;
    const loop = () => {
      if (!anim) return;
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(loop);
    };
    loop();

    const onResize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    return () => {
      anim = false;
      window.removeEventListener("resize", onResize);
      controls.dispose();
      renderer.dispose();
    };
  }, [url]);

  return (
    <div
      ref={mountRef}
      style={{
        width: "100%",
        height: 420,
        borderRadius: 16,
        overflow: "hidden",
        border: "1px solid rgba(255,255,255,0.12)",
      }}
    />
  );
}
