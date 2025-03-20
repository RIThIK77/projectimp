function toggleFields(select) {
    const userFields = document.getElementById('user-fields');
    userFields.style.display = select.value === 'user' ? 'block' : 'none';
}