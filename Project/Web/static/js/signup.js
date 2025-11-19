const signupForm = document.getElementById("signupForm");
const dialog = document.getElementById("dialog");
const togglePassword = document.getElementById("togglePassword");
const passwordInput = document.getElementById("password");
const emailInput = document.getElementById("email");
const termsCheckbox = document.getElementById("terms");
const termsError = document.getElementById("termsError");

// Lưu placeholder gốc
const emailPlaceholder = emailInput.placeholder;
const passwordPlaceholder = passwordInput.placeholder;

// Toggle show/hide password
togglePassword.addEventListener("click", () => {
  passwordInput.type = passwordInput.type === "password" ? "text" : "password";
});

// Handle Sign Up
signupForm.addEventListener("submit", (e) => {
  e.preventDefault();

  const email = emailInput.value.trim();
  const password = passwordInput.value.trim();
  const terms = termsCheckbox.checked;

  let valid = true;

  // Kiểm tra email
  if (!email) {
    emailInput.value = "";
    emailInput.placeholder = "Hãy điền vào trường email";
    emailInput.classList.add("error-input");
    valid = false;
  } else {
    emailInput.placeholder = emailPlaceholder;
    emailInput.classList.remove("error-input");
  }

  // Kiểm tra password
  if (!password) {
    passwordInput.value = "";
    passwordInput.placeholder = "Hãy điền vào trường mật khẩu";
    passwordInput.classList.add("error-input");
    valid = false;
  } else {
    passwordInput.placeholder = passwordPlaceholder;
    passwordInput.classList.remove("error-input");
  }

  // Kiểm tra terms
  if (!terms) {
    termsError.classList.remove("hidden");
    valid = false;
  } else {
    termsError.classList.add("hidden");
  }

  if (!valid) return;

  // Hiển thị dialog thành công
  dialog.classList.remove("hidden");

  // Tự động redirect sang Login sau 2 giây
  setTimeout(() => {
    window.location.href = "/login";
  }, 2000);
});
