const loginForm = document.getElementById("loginForm");
const emailInput = document.getElementById("email");
const passwordInput = document.getElementById("password");
const togglePassword = document.getElementById("togglePassword");
const dialog = document.getElementById("dialog");

const emailPlaceholder = emailInput.placeholder;
const passwordPlaceholder = passwordInput.placeholder;

// Toggle password visibility
togglePassword.addEventListener("click", () => {
  passwordInput.type = passwordInput.type === "password" ? "text" : "password";
});

// Khi người dùng bắt đầu nhập, xóa lỗi
emailInput.addEventListener("input", () => {
  emailInput.classList.remove("error");
  emailInput.placeholder = emailPlaceholder;
});

passwordInput.addEventListener("input", () => {
  passwordInput.classList.remove("error");
  passwordInput.placeholder = passwordPlaceholder;
});

// Handle Login
loginForm.addEventListener("submit", (e) => {
  e.preventDefault();
  let valid = true;

  if (!emailInput.value.trim()) {
    emailInput.value = "";
    emailInput.placeholder = "Hãy điền vào trường này";
    emailInput.classList.add("error");
    valid = false;
  }

  if (!passwordInput.value.trim()) {
    passwordInput.value = "";
    passwordInput.placeholder = "Hãy điền vào trường này";
    passwordInput.classList.add("error");
    valid = false;
  }

  if (!valid) return;

  // Hiển thị dialog thành công
  dialog.classList.remove("hidden");
  setTimeout(() => {
    window.location.href = "/information"; // chuyển trang
  }, 2000);
});
