const form = document.getElementById('bpForm');
const summaryBox = document.getElementById('summaryBox');
const formError = document.getElementById('formError');
const previewBtn = document.getElementById('previewBtn');
const downloadBtn = document.getElementById('downloadBtn');
const weightEl = document.getElementById('weight');
const heightEl = document.getElementById('height');
const bmiEl = document.getElementById('bmi');
const dobEl = document.getElementById('dob');
const ageEl = document.getElementById('age');
const genderEl = document.getElementById('gender');
const consentEl = document.getElementById('consent');
const restMinEl = document.getElementById('restMin');

function gatherData() {
  return {
    fullName: form.fullName.value.trim(),
    age: Number(form.age.value) || null,
    gender: form.gender.value,
    restMin: Number(form.restMin.value) || 5,
    weight: Number(form.weight.value) || null,
    height: Number(form.height.value) || null,
    bmi: Number(form.bmi.value) || null,
    consent: consentEl.checked
  };
}

function calculateBMI() {
  const w = parseFloat(weightEl.value);
  const h = parseFloat(heightEl.value);
  if (!isFinite(w) || !isFinite(h) || h <= 0) {
    bmiEl.value = '';
    return;
  }
  const m = h / 100;
  const bmi = w / (m * m);
  bmiEl.value = (Math.round(bmi * 10) / 10).toString();
}

function calculateAge() {
  const dob = new Date(dobEl.value);
  if (isNaN(dob)) return;
  const today = new Date();
  let age = today.getFullYear() - dob.getFullYear();
  const m = today.getMonth() - dob.getMonth();
  if (m < 0 || (m === 0 && today.getDate() < dob.getDate())) age--;
  ageEl.value = age;
}

function updateSummary(data) {
  const parts = [];
  if (data.fullName) parts.push(`<strong>${escapeHtml(data.fullName)}</strong>`);
  if (data.age) parts.push(`<strong>Tuổi:</strong> ${escapeHtml(data.age)}`);
  if (data.gender) parts.push(`<strong>Giới tính:</strong> ${escapeHtml(data.gender)}`);
  if (data.restMin) parts.push(`<strong>Nghỉ:</strong> ${escapeHtml(data.restMin)} phút`);
  if (data.weight) parts.push(`<strong>Cân nặng:</strong> ${escapeHtml(data.weight)} kg`);
  if (data.height) parts.push(`<strong>Chiều cao:</strong> ${escapeHtml(data.height)} cm`);
  if (data.bmi) parts.push(`<strong>BMI:</strong> ${escapeHtml(data.bmi)}`);
  summaryBox.innerHTML = parts.length ? parts.join('<br>') : 'Chưa có thông tin.';
}

function escapeHtml(s) {
  if (!s) return '';
  return String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

function validate(data) {
  const errs = [];
  if (!data.fullName) errs.push('Vui lòng nhập họ và tên.');
  if (!data.age || data.age < 0) errs.push('Tuổi không hợp lệ.');
  if (!data.weight || data.weight <= 0) errs.push('Cân nặng không hợp lệ.');
  if (!data.height || data.height <= 0) errs.push('Chiều cao không hợp lệ.');
  if (!data.consent) errs.push('Bạn cần đồng ý để tiếp tục.');
  if (!data.restMin || data.restMin < 0) errs.push('Thời gian nghỉ không hợp lệ.');
  return errs;
}

['input', 'change'].forEach(ev => {
  form.addEventListener(ev, () => {
    calculateBMI();
    calculateAge();
    updateSummary(gatherData());
  });
});

form.addEventListener('submit', e => {
  e.preventDefault();
  const data = gatherData();
  const errs = validate(data);
  formError.textContent = errs.length ? errs.join(' ') : '';
  updateSummary(data);
  if (!errs.length) {
    fetch("/information", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    }).then(res => {
      if (!res.ok) throw new Error("Không lưu được thông tin");
      window.location.href = "/measure";
    }).catch(err => { formError.textContent = err.message; });
  }
});

previewBtn.addEventListener('click', () => {
  const data = gatherData();
  const errs = validate(data);
  formError.textContent = errs.length ? errs.join(' ') : '';
  updateSummary(data);
  if (!errs.length) {
    const w = window.open('', '_blank', 'width=600,height=500');
    w.document.write('<pre style="white-space:pre-wrap;font-family:monospace;padding:12px">' + escapeHtml(JSON.stringify(data, null, 2)) + '</pre>');
    w.document.title = 'Xem trước thông tin';
  }
});

downloadBtn.addEventListener('click', () => {
  const data = gatherData();
  const errs = validate(data);
  if (errs.length) { formError.textContent = errs.join(' '); return; }
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `bp-info-${(data.fullName || 'user').replaceAll(' ', '_')}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

updateSummary(gatherData());
