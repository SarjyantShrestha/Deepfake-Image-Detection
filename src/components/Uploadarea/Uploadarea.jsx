import React, { useState } from "react";
import "./Uploadarea.css";

function Uploadarea() {
  // const [classificationResult1, setClassificationResult1] = useState(null);
  const [classificationResult2, setClassificationResult2] = useState(null);
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
      // setClassificationResult1(result.prediction);
      setClassificationResult2(result.prediction2);
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
    // setClassificationResult1(null);
    setClassificationResult2(null);
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
        <label htmlFor="file-upload" className="custom-file-input">
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
      {classificationResult2 !== null && (
        <p className="result">
          {/* Classification result (old): <span>{classificationResult1}</span>
          <br /> */}
          Classification result: <span>{classificationResult2}</span>
        </p>
      )}
    </div>
  );
}

export default Uploadarea;
