import { useEffect, useRef } from "react";
import * as THREE from "three";

export default function FallbackBox({ title = "3D Preview", subtitle = "" }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current) return;

    const width = ref.current.clientWidth;
    const height = ref.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1020);

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 100);
    camera.position.set(2, 2, 2);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    ref.current.appendChild(renderer.domElement);

    // lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(4, 6, 4);
    scene.add(dir);

    // box
    const geom = new THREE.BoxGeometry(1, 1, 1);
    const mat = new THREE.MeshStandardMaterial({ color: 0xffd54a, metalness: 0.1, roughness: 0.6 });
    const cube = new THREE.Mesh(geom, mat);
    scene.add(cube);

    const grid = new THREE.GridHelper(10, 10, 0x223047, 0x223047);
    grid.position.y = -0.55;
    scene.add(grid);

    let raf = 0;
    const animate = () => {
      cube.rotation.y += 0.01;
      cube.rotation.x += 0.005;
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    animate();

    const onResize = () => {
      if (!ref.current) return;
      const w = ref.current.clientWidth;
      const h = ref.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      geom.dispose();
      mat.dispose();
      if (renderer.domElement && renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div style={{ position: "relative", width: "100%", height: 420, borderRadius: 16, overflow: "hidden", border: "1px solid rgba(255,255,255,0.12)" }}>
      <div ref={ref} style={{ width: "100%", height: "100%" }} />
      <div style={{ position: "absolute", top: 10, left: 12, color: "white", fontSize: 12, opacity: 0.85 }}>
        <div style={{ fontWeight: 700 }}>{title}</div>
        {subtitle ? <div style={{ opacity: 0.8 }}>{subtitle}</div> : null}
      </div>
    </div>
  );
}
