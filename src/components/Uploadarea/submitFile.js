// submitFile.js
const submitFile = async (inpFile) => {
  const endpoint = "http://127.0.0.1:8000/upload";
  const formData = new FormData();
  formData.append("inpFile", inpFile);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    console.log("Response:", data);
  } catch (error) {
    console.error("Error:", error);
  }
};

export default submitFile;
