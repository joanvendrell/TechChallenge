const API_BASE = "http://localhost:8000";

export async function createJob(payload) {
  const res = await fetch(`${API_BASE}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json(); // { job_id, status, message }
}

export async function getJob(jobId) {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json(); // JobDetail
}

export async function cancelJob(job_id) {
  const res = await fetch(`http://127.0.0.1:8000/jobs/${job_id}/cancel`, {
    method: "POST",
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`Cancel failed (${res.status}): ${txt}`);
  }
  return res.json().catch(() => ({}));
}
