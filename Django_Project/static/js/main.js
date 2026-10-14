/**
 * Graze Smasher Product Hunting Tool - Core JavaScript
 * Vanilla JavaScript (No React, Vue, or external dependencies except Bootstrap bundle)
 */

document.addEventListener('DOMContentLoaded', function () {
    'use strict';

    // 1. Initialize Bootstrap Tooltips & Popovers
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.forEach(function (tooltipTriggerEl) {
        new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // 2. Auto-Dismiss Alert Messages after 6 seconds
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(function (alertEl) {
        setTimeout(function () {
            const bsAlert = bootstrap.Alert.getInstance(alertEl) || new bootstrap.Alert(alertEl);
            if (bsAlert && alertEl.classList.contains('show')) {
                bsAlert.close();
            }
        }, 6000);
    });

    // 3. Global Confirmation Modal Handler
    const confirmModalEl = document.getElementById('confirmationModal');
    if (confirmModalEl) {
        let targetFormOrUrl = null;
        let isFormSubmit = false;

        document.querySelectorAll('[data-confirm]').forEach(function (trigger) {
            trigger.addEventListener('click', function (e) {
                e.preventDefault();
                const msg = trigger.getAttribute('data-confirm') || 'Are you sure you want to proceed?';
                const modalBody = document.getElementById('confirmationModalBody');
                if (modalBody) {
                    modalBody.textContent = msg;
                }

                if (trigger.tagName.toLowerCase() === 'button' && trigger.closest('form')) {
                    targetFormOrUrl = trigger.closest('form');
                    isFormSubmit = true;
                } else {
                    targetFormOrUrl = trigger.getAttribute('href');
                    isFormSubmit = false;
                }

                const modal = new bootstrap.Modal(confirmModalEl);
                modal.show();
            });
        });

        const confirmBtn = document.getElementById('confirmActionBtn');
        if (confirmBtn) {
            confirmBtn.addEventListener('click', function () {
                if (isFormSubmit && targetFormOrUrl) {
                    targetFormOrUrl.submit();
                } else if (targetFormOrUrl) {
                    window.location.href = targetFormOrUrl;
                }
            });
        }
    }

    // 4. Smooth Score Progress Bar Fill Animation
    const progressBars = document.querySelectorAll('.progress-bar[aria-valuenow]');
    progressBars.forEach(function (bar) {
        const val = bar.getAttribute('aria-valuenow');
        if (val) {
            bar.style.transition = 'width 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
            bar.style.width = val + '%';
        }
    });

    // 5. Search Bar Quick Reset Helper
    const searchInputs = document.querySelectorAll('input[type="text"][name="q"]');
    searchInputs.forEach(function (input) {
        input.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                input.value = '';
                const form = input.closest('form');
                if (form) form.submit();
            }
        });
    });
});

