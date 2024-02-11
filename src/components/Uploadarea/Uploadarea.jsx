import React, { useState } from "react";
import "./Uploadarea.css";

function Uploadarea() {
  const [classificationResult, setClassificationResult] = useState();

  const upload = async (formData) => {
    try {
      const response = await fetch("http://127.0.0.1:8000/uploadfile", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      console.log("Success:", result);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    const fileField = document.querySelector('input[type="file"]');
    formData.append("file", fileField.files[0]);
    await upload(formData);
  };

  return (
    <form className="form" onSubmit={handleSubmit}>
      <input type="file" /> <br />
      <button type="submit">Upload File</button>
    </form>
  );
}

export default Uploadarea;
