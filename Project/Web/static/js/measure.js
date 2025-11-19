document.addEventListener("DOMContentLoaded", () => {
  const chooseCamera = document.getElementById("chooseCamera");
  const chooseUpload = document.getElementById("chooseUpload");
  const cameraSection = document.getElementById("cameraSection");
  const uploadSection = document.getElementById("uploadSection");
  const resultSection = document.getElementById("resultSection");
  const sbpSpan = document.getElementById("sbp");
  const dbpSpan = document.getElementById("dbp");
  const previewVideo = document.getElementById("previewVideo");
  const statusMessage = document.getElementById("statusMessage");
  const webcamStream = document.getElementById("webcamStream");
  const optionsDiv = document.querySelector(".options");

  let webcamActive = false;
  const startMeasure = document.createElement("button");
  startMeasure.textContent = "Bắt đầu đo";
  startMeasure.className = "option-btn hidden";
  cameraSection.appendChild(startMeasure);

  function createGauge(id,value,min,max,color){
    return new Chart(document.getElementById(id),{
      type:'doughnut',
      data:{datasets:[{data:[value-min,max-value],backgroundColor:[color,'#eee'],borderWidth:0}]},
      options:{cutout:'70%',rotation:-90,circumference:180,plugins:{tooltip:{enabled:false},legend:{display:false}}},
      plugins:[{id:'text',beforeDraw:(chart)=>{const ctx=chart.ctx;const{width}=chart;ctx.restore();ctx.font="bold 18px Arial";ctx.fillStyle="#333";ctx.textBaseline="middle";const text=value.toFixed(0);const textX=Math.round((width-ctx.measureText(text).width)/2);const textY=chart._metasets[0].data[0].y+40;ctx.fillText(text,textX,textY);ctx.save();}}]
    });
  }

  // Quay video
  chooseCamera.addEventListener("click",()=>{
    cameraSection.classList.remove("hidden");
    uploadSection.classList.add("hidden");
    resultSection.classList.add("hidden");
    optionsDiv.classList.add("hidden");
    statusMessage.textContent="💡 Webcam sẵn sàng. Nhấn 'Bắt đầu đo' để đo huyết áp.";
    if(!webcamActive){ webcamStream.src="/video_feed"; webcamActive=true; }
    startMeasure.classList.remove("hidden");
  });

  startMeasure.addEventListener("click",()=>{
    statusMessage.textContent="⏳ Đang đo huyết áp... Vui lòng giữ yên";
    startMeasure.classList.add("hidden");

    fetch("/webcam").then(res=>res.json()).then(data=>{
      if(data.error){ statusMessage.textContent="❌ "+data.error; return; }

      cameraSection.classList.add("hidden");
      uploadSection.classList.add("hidden");
      resultSection.classList.remove("hidden");

      sbpSpan.textContent=data.sbp?.toFixed(2)||"N/A";
      dbpSpan.textContent=data.dbp?.toFixed(2)||"N/A";

      createGauge("sbpGauge",data.sbp||0,80,180,"green");
      createGauge("dbpGauge",data.dbp||0,50,120,"orange");

      statusMessage.textContent="✅ Đo xong!";
    }).catch(err=>{ statusMessage.textContent="❌ Lỗi xử lý webcam"; console.error(err); });
  });

  // Upload
  chooseUpload.addEventListener("click",()=>{
    uploadSection.classList.remove("hidden");
    cameraSection.classList.add("hidden");
    resultSection.classList.add("hidden");
    optionsDiv.classList.add("hidden");
    startMeasure.classList.add("hidden");
    if(webcamActive){ webcamStream.src=""; webcamActive=false; }
  });

  document.getElementById("videoInput").addEventListener("change",(e)=>{
    const file=e.target.files[0];
    if(file){ previewVideo.src=URL.createObjectURL(file); previewVideo.classList.remove("hidden"); }
  });

  document.getElementById("uploadForm").addEventListener("submit",(e)=>{
    e.preventDefault();
    const formData=new FormData(e.target);
    statusMessage.textContent="⏳ Đang xử lý video...";

    fetch("/upload",{method:"POST",body:formData}).then(res=>res.json()).then(data=>{
      if(data.error){ statusMessage.textContent="❌ "+data.error; return; }

      uploadSection.classList.add("hidden");
      previewVideo.classList.add("hidden");
      cameraSection.classList.add("hidden");
      resultSection.classList.remove("hidden");

      sbpSpan.textContent=data.sbp?.toFixed(2)||"N/A";
      dbpSpan.textContent=data.dbp?.toFixed(2)||"N/A";

      createGauge("sbpGauge",data.sbp||0,80,180,"green");
      createGauge("dbpGauge",data.dbp||0,50,120,"orange");

      statusMessage.textContent="✅ Xử lý xong!";
    }).catch(err=>{ statusMessage.textContent="❌ Lỗi upload video"; console.error(err); });
  });

  // Đo lại
  document.getElementById("restart").addEventListener("click",()=>{
    resultSection.classList.add("hidden");
    cameraSection.classList.add("hidden");
    uploadSection.classList.add("hidden");
    optionsDiv.classList.remove("hidden");
    statusMessage.textContent="";
    startMeasure.classList.add("hidden");
    if(webcamActive){ webcamStream.src=""; webcamActive=false; }
  });
});
