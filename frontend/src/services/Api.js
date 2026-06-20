const BASE_URL = "https://soumadhut-ai-doc-rare-disease-api.hf.space";

export const predictDisease = async (symptoms, image, top_k = 5) => {
  const formData = new FormData();
  formData.append("symptoms", symptoms);
  formData.append("top_k", top_k);

  if (image) {
    formData.append("image", image);
    const res = await fetch(`${BASE_URL}/predict`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
    return await res.json();
  }

  const res = await fetch(`${BASE_URL}/predict/text`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
  return await res.json();
};

export const getAnalytics = async () => {
  const res = await fetch(`${BASE_URL}/analytics`);
  if (!res.ok) throw new Error(`Analytics failed: ${res.status}`);
  return await res.json();
};

export const getHistory = async () => {
  const res = await fetch(`${BASE_URL}/history`);
  if (!res.ok) throw new Error(`History failed: ${res.status}`);
  return await res.json();
};