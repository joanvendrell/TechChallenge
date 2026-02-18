import { useEffect, useState } from "react";

function extOf(url) {
  try {
    const u = new URL(url, window.location.href);
    const p = u.pathname.toLowerCase();
    const i = p.lastIndexOf(".");
    return i >= 0 ? p.slice(i + 1) : "";
  } catch {
    const p = String(url).toLowerCase();
    const i = p.lastIndexOf(".");
    return i >= 0 ? p.slice(i + 1) : "";
  }
}

export default function VtkProbe({ url }) {
  const [status, setStatus] = useState("idle"); // idle | ok | skip | fail
  const [info, setInfo] = useState("");

  useEffect(() => {
    if (!url) return;

    let cancelled = false;
    setStatus("idle");
    setInfo("");

    const ext = extOf(url);
    const isVtkXml = ext === "vtu" || ext === "pvtu" || ext === "vtp";

    // If it's not a VTK-XML artifact, don't treat as error
    if (!isVtkXml) {
      setStatus("skip");
      setInfo(`Skipping VTK probe for .${ext || "unknown"} (not VTK-XML).`);
      return () => {
        cancelled = true;
      };
    }

    fetch(url)
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const buf = await r.arrayBuffer();

        // VTK XML should contain <VTKFile ...>
        const head = new TextDecoder().decode(buf.slice(0, 256));
        const headLower = head.toLowerCase();

        if (!head.includes("<")) throw new Error("Not XML (first bytes not '<')");
        if (!headLower.includes("vtkfile")) {
          throw new Error("Missing VTKFile header (not VTK XML?)");
        }

        if (!cancelled) {
          setStatus("ok");
          setInfo(`Fetched ${buf.byteLength} bytes; header looks like VTK XML (.${ext}).`);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setStatus("fail");
          setInfo(e?.message ? e.message : String(e));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [url]);

  const color =
    status === "ok"
      ? "rgba(150,255,150,0.9)"
      : status === "skip"
      ? "rgba(255,255,255,0.7)"
      : status === "fail"
      ? "rgba(255,150,150,0.9)"
      : "rgba(255,255,255,0.8)";

  return (
    <div style={{ marginTop: 10, fontSize: 12, color }}>
      <b>VTK probe:</b>{" "}
      {status === "idle"
        ? "loading..."
        : status === "ok"
        ? `OK — ${info}`
        : status === "skip"
        ? info
        : `FAILED — ${info}`}
    </div>
  );
}
