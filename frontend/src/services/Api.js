import { auth } from "./firebase";

const BASE_URL = "https://soumadhut-ai-doc-rare-disease-api.hf.space";

/** Attaches the current user's Firebase ID token, if logged in. */
async function authHeaders() {
  const user = auth.currentUser;
  if (!user) return {};
  const token = await user.getIdToken();
  return { Authorization: `Bearer ${token}` };
}

export const predictDisease = async (symptoms, image, top_k = 5) => {
  const formData = new FormData();
  formData.append("symptoms", symptoms);
  formData.append("top_k", top_k);

  const headers = await authHeaders();

  if (image) {
    formData.append("image", image);
    const res = await fetch(`${BASE_URL}/predict`, { method: "POST", body: formData, headers });
    if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
    return await res.json();
  }

  const res = await fetch(`${BASE_URL}/predict/text`, { method: "POST", body: formData, headers });
  if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
  return await res.json();
};

export const getAnalytics = async () => {
  const headers = await authHeaders();
  const res = await fetch(`${BASE_URL}/analytics`, { headers });
  if (!res.ok) throw new Error(`Analytics failed: ${res.status}`);
  return await res.json();
};

export const getHistory = async () => {
  const headers = await authHeaders();
  const res = await fetch(`${BASE_URL}/history`, { headers });
  if (!res.ok) throw new Error(`History failed: ${res.status}`);
  return await res.json();
};