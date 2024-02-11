import React, { useState } from "react";
import "./Uploadarea.css";

function Uploadarea() {
  const [classificationResult, setClassificationResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [filename, setFilename] = useState("No image chosen");
  const [imageURL, setImageURL] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setImageURL(url);
      setFilename(file.name);
    } else {
      setImageURL(null);
      setFilename("No file chosen");
    }
  };

  const upload = async (formData) => {
    setLoading(true); //setLoading is true when starting the request
    try {
      const response = await fetch("http://127.0.0.1:8000/uploadfile", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      console.log("Success:", result);
      setClassificationResult(result.prediction);
      // setTimeout(() => {
      //   setClassificationResult(null);
      // }, 4000); //reset after 3 seconds
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    setClassificationResult(null);
    e.preventDefault();
    const formData = new FormData();
    const fileField = document.querySelector('input[type="file"]');
    formData.append("file", fileField.files[0]);
    await upload(formData);
  };

  return (
    <div className="maindiv">
      <form className="form" onSubmit={handleSubmit}>
        {/* <input className="choose-file" type="file" /> <br /> */}
        <label for="file-upload" class="custom-file-input">
          Choose an image
        </label>
        {imageURL && (
          <img className="uploaded-img" src={imageURL} alt="Selected Image" />
        )}
        <p id="filename">{filename}</p>
        <input
          id="file-upload"
          className="choose-file"
          type="file"
          onChange={handleFileChange} // Call handleFileChange when the file input changes
        />
        <button type="submit">Upload image</button>
      </form>
      {loading && <p className="result">Processing..</p>}
      {classificationResult !== null && (
        <p className="result">
          Classification result: <span>{classificationResult}</span>
        </p>
      )}
    </div>
  );
}

export default Uploadarea;
