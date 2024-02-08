import React from "react";
import "./Uploadarea.css";
import submitFile from "./submitFile"; // Import the submitFile function

function Uploadarea() {
  const handleSubmit = (e) => {
    e.preventDefault();
    const inpFile = document.getElementById("inpFile");
    submitFile(inpFile.files[0]); // Call the submitFile function
  };

  return (
    <form className="form" id="myForm" onSubmit={handleSubmit}>
      <input type="file" id="inpFile" /> <br />
      <button type="submit"> Testing</button>
    </form>
  );
}

export default Uploadarea;
