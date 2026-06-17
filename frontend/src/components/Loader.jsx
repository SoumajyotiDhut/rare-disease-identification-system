function Loader({ message = "Loading…" }) {
    return (
        <div style={{
            minHeight: "100vh",
            background: "#F8FAFB",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "column",
            gap: 20,
            fontFamily: "'Inter', sans-serif",
        }}>
            {/* Animated ring */}
            <div style={{ position: "relative", width: 56, height: 56 }}>
                <div style={{
                    position: "absolute", inset: 0,
                    border: "3px solid #E8EFF5",
                    borderTop: "3px solid #0B7B6F",
                    borderRadius: "50%",
                    animation: "spin 0.9s linear infinite",
                }} />
                <div style={{
                    position: "absolute", inset: 8,
                    border: "2px solid #F0F5F8",
                    borderTop: "2px solid #1D6FA4",
                    borderRadius: "50%",
                    animation: "spin 1.4s linear infinite reverse",
                }} />
            </div>

            <div style={{ textAlign: "center" }}>
                <p style={{ fontSize: 15, color: "#5A7184", fontWeight: 500, margin: "0 0 4px" }}>{message}</p>
                <p style={{ fontSize: 12, color: "#9BB8CC", margin: 0 }}>AI DOC · Rare Disease Assistant</p>
            </div>

            <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
        </div>
    );
}

export default Loader;