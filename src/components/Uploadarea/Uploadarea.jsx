import React, { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import "./Uploadarea.css";

function Uploadarea() {
  const onDrop = useCallback((acceptedFiles) => {
    // Do something with the files
  }, []);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  return (
    <div className="uploadarea">
      <div className="uploadinput" {...getRootProps()}>
        <input {...getInputProps()} />
        {isDragActive ? (
          <p>Drop the files here ...</p>
        ) : (
          <p>Drag 'n' drop some files here, or click to select files</p>
        )}
      </div>
      <div className="checkbutton">
        <button>Check image</button>
      </div>
    </div>
  );
}

export default Uploadarea;
