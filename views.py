from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from .forms import UserRegisterForm, ProfileUpdateForm, EmployeeForm
from .models import *
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from django.db import transaction
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.contrib.auth.decorators import user_passes_test

def superuser_check(user):
    if not user.is_superuser:
        messages.error(user, "Access Denied! Only admins can view this page.")
    return user.is_superuser

# User login view

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)  # Log in the user

            # Redirect based on user type
            if user.is_superuser:
                return redirect('admin-dashboard')  # Redirect superuser to admin dashboard
            else:
                return redirect('employee-dashboard')  # Redirect normal user to profile page
    else:
        form = AuthenticationForm()
    
    return render(request, 'account/login.html', {'form': form})

# User logout view
def user_logout(request):
    logout(request)
    return redirect('login')  # Redirect to login page after logout


# User registration view
def register(request):
    if request.method == 'POST':
        user_form = UserRegisterForm(request.POST)
        if user_form.is_valid():
            user = user_form.save()
            Profile.objects.create(user=user)  # Create profile for new user
            return redirect('login')  # Redirect to login after registration
    else:
        user_form = UserRegisterForm()
    
    return render(request, 'account/register.html', {'user_form': user_form})

from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import EmployeeForm
from .models import Employee, Profile
from .face_recognition_utils import extract_face_encoding  # Utility function to extract face encoding
import logging

# Set up logging for debugging
logger = logging.getLogger(__name__)

@login_required(login_url='login')
def user_profile(request):
    user = request.user

    # Fetch or create the Employee & Profile objects
    employee, created = Employee.objects.get_or_create(user=user)
    profile, created = Profile.objects.get_or_create(user=user)

    title = "My Profile"

    if request.method == "POST":
        form = EmployeeForm(request.POST, request.FILES, instance=employee)

        if form.is_valid():
            # Update User model
            user.username = form.cleaned_data['username']
            user.first_name = form.cleaned_data['first_name']
            user.last_name = form.cleaned_data['last_name']
            user.email = form.cleaned_data['email']
            user.save()

            # Update Profile model
            profile.contact = form.cleaned_data['contact']
            if 'profile_image' in request.FILES:
                profile.profile_image = request.FILES['profile_image']  # Save new profile image
                profile.save()

                try:
                    # Extract face encoding from the new profile image
                    image_path = profile.profile_image.path
                    face_encoding = extract_face_encoding(image_path)
                    if face_encoding is not None:
                        profile.set_encoding(face_encoding)  # Save the encoding to the profile
                        profile.save()  # Make sure to save after updating the encoding
                        logger.debug(f"Face encoding updated successfully for {profile.user.username}")
                    else:
                        logger.warning("No face encoding found in the image.")
                except Exception as e:
                    logger.error(f"Error during face recognition: {str(e)}")
                    messages.error(request, f"Error during face recognition: {str(e)}")

            # Update Employee model
            employee = form.save(commit=False)
            employee.user = user  # Ensure correct user assignment
            employee.save()

            messages.success(request, "Profile updated successfully!")
            return redirect('profile')  # Redirect back to the profile page

        else:
            messages.error(request, "Please correct the errors below.")  # Show form errors

    else:
        # Pre-fill form with existing data
        form = EmployeeForm(instance=employee, initial={
            'username': user.username,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'email': user.email,
            'contact': profile.contact,
        })

    if request.user.is_superuser:
        return render(request, 'admin/profile.html', {'form': form, 'employee': employee, 'title': title})
    else:
        return render(request, 'employee/profile.html', {'form': form, 'employee': employee, 'title': title})


from django.utils import timezone
from django.shortcuts import render
from django.contrib.auth.decorators import user_passes_test
from django.shortcuts import render
from django.utils import timezone
from datetime import timedelta
from .models import Employee, Attendance


@user_passes_test(lambda user: user.is_superuser, login_url='login')
def admin_dashboard(request):
    today = timezone.now().date()
    
    # Active employees count
    total_employees = Employee.objects.filter(status="Active").count()

    # Attendance counts
    present = Attendance.objects.filter(date=today, status="Present").count()
    absent = Attendance.objects.filter(date=today, status="Absent").count()
    not_reported = total_employees - (present + absent)

    # Last 7 days attendance for bar chart
    week_dates = [today - timedelta(days=i) for i in range(6, -1, -1)]
    week_labels = [date.strftime("%a") for date in week_dates]  # Labels (Mon, Tue, ...)
    week_present = [
        Attendance.objects.filter(date=date, status="Present").count()
        for date in week_dates
    ]
    week_absent = [
        Attendance.objects.filter(date=date, status="Absent").count()
        for date in week_dates
    ]
    week_not_reported = [
        total_employees - (week_present[i] + week_absent[i])
        for i in range(7)
    ]

    # Last 4 weeks attendance for area chart (Only Mondays)
     # 🔹 Last 4 weeks attendance (Area Chart)
    month_labels = []
    month_present = []
    month_absent = []
    month_not_reported = []

    for i in range(3, -1, -1):  # Last 4 weeks (current week included)
        start_date = today - timedelta(weeks=i, days=today.weekday())  # Start of the week (Monday)
        end_date = start_date + timedelta(days=6)  # End of the week (Sunday)

        # Sum total attendance within that week
        week_present_total = Attendance.objects.filter(date__range=[start_date, end_date], status="Present").count()
        week_absent_total = Attendance.objects.filter(date__range=[start_date, end_date], status="Absent").count()
        week_not_reported_total = total_employees - (week_present_total + week_absent_total)

        month_labels.append(f"Week {i+1}")  # Labels: "Week 1", "Week 2", ...
        month_present.append(week_present_total)
        month_absent.append(week_absent_total)
        month_not_reported.append(week_not_reported_total)

    # 🔹 Recent 4 attendance records (with time difference)
    recent_attendance = Attendance.objects.order_by('-timestamp')[:5]
    recent_attendance_data = [
        {
            'name': f"{att.emp.user.first_name} {att.emp.user.last_name}",
            'time_ago': timesince(att.timestamp) + " ago"
        }
        for att in recent_attendance
    ]

    # 🔹 Recent 2 leave requests
    recent_leaves = Leaves.objects.order_by('-timestamp')[:3]
    recent_leave_data = [
        {
            'name': f"{leave.emp.user.first_name} {leave.emp.user.last_name}",
            # 'leave_type': leave.leave_type,
            'reason': leave.reason,
            'time_ago': timesince(leave.timestamp) + " ago"
        }
        for leave in recent_leaves
    ]

    context = {
        'total_employees': total_employees,
        'present': present,
        'absent': absent,
        'not_reported': not_reported,
        'week_labels': week_labels,
        'week_present': week_present,
        'week_absent': week_absent,
        'week_not_reported': week_not_reported,
        'month_labels': month_labels,
        'month_present': month_present,
        'month_absent': month_absent,
        'month_not_reported': month_not_reported,
        'recent_attendance': recent_attendance_data,
        'recent_leaves': recent_leave_data
    }

    return render(request, 'admin/dashboard.html', context)

@user_passes_test(lambda user: user.is_superuser, login_url='login')
def employees(request):
    departments = Department.objects.all()  # Get all departments
    department_id = request.GET.get('department')  # Get selected department

    print(f"Selected Department ID: {department_id}")  # Debugging

    # If "All" is selected or no department is selected, show all employees
    if department_id and department_id.isdigit():  
        employees = Employee.objects.filter(dep_id=department_id)
    else:
        employees = Employee.objects.all()  # Show all employees when "All" is selected

    return render(request, 'admin/employees.html', {
        'employees': employees,
        'departments': departments,
        'selected_department': department_id  # Pass selected department for UI update
    })

from django.db import transaction, IntegrityError
from django.contrib import messages
from django.shortcuts import render, redirect
from .forms import EmployeeForm
from .models import Employee, Profile, User

@user_passes_test(lambda user: user.is_superuser, login_url='login')
def add_employee(request):
    title = "Add New Employee"
    
    if request.method == "POST":
        form = EmployeeForm(request.POST, request.FILES)  # Handle file uploads (profile image)
        
        if form.is_valid():
            try:
                with transaction.atomic():  # Ensures all objects are created together
                    # Create User
                    user = User.objects.create_user(
                        username=form.cleaned_data['username'],
                        first_name=form.cleaned_data['first_name'],
                        last_name=form.cleaned_data['last_name'],
                        email=form.cleaned_data['email'],
                        password=form.cleaned_data['password']
                    )

                    # Create Profile
                    profile = Profile.objects.create(
                        user=user,
                        profile_image=form.cleaned_data['profile_image'],
                        contact=form.cleaned_data['contact']
                    )

                    # Extract face encoding from the uploaded profile image
                    try:
                        image_path = profile.profile_image.path
                        face_encoding = extract_face_encoding(image_path)

                        if face_encoding is not None:
                            profile.set_encoding(face_encoding)  # Save encoding
                            profile.save()
                        else:
                            messages.warning(request, "No face encoding found in the image.")
                    
                    except Exception as e:
                        messages.warning(request, f"Face recognition error: {str(e)}")
                        logger.error(f"Face recognition error: {str(e)}")

                    # Create Employee
                    employee = Employee.objects.create(
                        user=user,
                        dep=form.cleaned_data['dep'],
                        role=form.cleaned_data['role'],
                        status=form.cleaned_data['status']
                    )

                messages.success(request, "Employee added successfully!")
                return redirect('employees')  # Redirect to employee list page

            except IntegrityError:
                messages.error(request, "Username already exists. Please choose a different username.")

    else:
        form = EmployeeForm()
    
    return render(request, 'admin/add_employee.html', {'form': form, 'title': title})

@user_passes_test(lambda user: user.is_superuser, login_url='login')
def delete_employee(request, employee_id):
    employee = get_object_or_404(Employee, pk=employee_id)
    user = employee.user  # Get associated user before deleting employee
    employee.delete()
    user.delete()  # Delete associated user
    messages.success(request, "Employee deleted successfully!")
    return redirect('employees')


from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from .models import Employee, Profile
from .forms import EmployeeForm
@user_passes_test(lambda user: user.is_superuser, login_url='login')
def update_employee(request, employee_id):
    employee = get_object_or_404(Employee, pk=employee_id)
    user = employee.user  # Associated User object
    profile, created = Profile.objects.get_or_create(user=user)  # Ensure profile exists
    title = f"Update Employee ({user.first_name} {user.last_name})"

    if request.method == "POST":
        form = EmployeeForm(request.POST, request.FILES, instance=employee)

        if form.is_valid():
            try:
                with transaction.atomic():  # Ensures all updates happen together
                    # Update User model
                    user.username = form.cleaned_data['username']
                    user.first_name = form.cleaned_data['first_name']
                    user.last_name = form.cleaned_data['last_name']
                    user.email = form.cleaned_data['email']
                    user.save()

                    # Update Profile model
                    profile.contact = form.cleaned_data['contact']
                    
                    if 'profile_image' in request.FILES:
                        profile.profile_image = request.FILES['profile_image']  # Save new profile image
                        profile.save()

                        # Extract face encoding from the new profile image
                        try:
                            image_path = profile.profile_image.path
                            face_encoding = extract_face_encoding(image_path)

                            if face_encoding is not None:
                                profile.set_encoding(face_encoding)  # Save encoding
                                profile.save()
                                logger.debug(f"Face encoding updated successfully for {profile.user.username}")
                            else:
                                messages.warning(request, "No face encoding found in the image.")
                        
                        except Exception as e:
                            messages.warning(request, f"Face recognition error: {str(e)}")
                            logger.error(f"Face recognition error: {str(e)}")

                    # Update Employee model
                    employee = form.save(commit=False)
                    employee.user = user  # Ensure correct user assignment
                    employee.save()

                    messages.success(request, "Employee updated successfully!")
                    return redirect('employees')  # Redirect to employee list page

            except IntegrityError:
                messages.error(request, "Username already exists. Please choose a different username.")

        else:
            print("Form Errors:", form.errors)
            messages.error(request, "Please correct the errors below.")  # Show form errors

    else:
        # Pre-fill form with existing data
        form = EmployeeForm(instance=employee, initial={
            'username': user.username,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'email': user.email,
            'password': user.password,
            'contact': profile.contact,
        })

    return render(request, 'admin/add_employee.html', {'form': form, 'employee': employee, 'title': title})

@user_passes_test(lambda user: user.is_superuser, login_url='login')
def departments(request):
    departments = Department.objects.all()  # Fetch all departments
    return render(request, 'admin/departments.html', {'departments': departments})


from .forms import DepartmentForm

@user_passes_test(lambda user: user.is_superuser, login_url='login')
def add_department(request):
    if request.method == "POST":
        form = DepartmentForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Department added successfully!")
            return redirect('departments')  # Redirect to department list view after adding
    else:
        form = DepartmentForm()
    
    return render(request, 'admin/addDepartment.html', {'form': form})

@user_passes_test(lambda user: user.is_superuser, login_url='login')
def update_department(request, department_id):
    department = get_object_or_404(Department, pk=department_id)

    if request.method == "POST":
        form = DepartmentForm(request.POST, instance=department)
        if form.is_valid():
            form.save()
            messages.success(request, "Department updated successfully!")
            return redirect("departments")  # Change to your department list view
        else:
            messages.error(request, "Error updating department. Please check the form.")
    else:
        form = DepartmentForm(instance=department)

    return render(request, "admin/update_department.html", {"form": form, "department": department})

@user_passes_test(lambda user: user.is_superuser, login_url='login')
def delete_department(request, department_id):
    department = get_object_or_404(Department, pk=department_id)
    department.delete()
    messages.success(request, "Department deleted successfully!")
    return redirect("departments")  # Redirect to department list after deletion


@user_passes_test(lambda user: user.is_superuser, login_url='login')
def attendance_list(request):
    # attendances = Attendance.objects.select_related('emp__user').order_by('-date')
    attendances = Attendance.objects.order_by('-date')
    return render(request, 'attendance/attendance_list.html', {'attendances': attendances})



import openpyxl
from django.http import HttpResponse
from .models import Attendance

def export_attendance(request):
    # Fetch all the attendance records
    attendances = Attendance.objects.all()

    # Create a new workbook and sheet
    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.title = "Attendance List"

    # Set the headers for the Excel file
    sheet.append(["Employee Name", "Department", "Date & Time", "Status"])

    # Add the attendance data
    for attendance in attendances:
        employee_name = f"{attendance.emp.user.first_name} {attendance.emp.user.last_name}"
        department = attendance.emp.dep.name
        timestamp = attendance.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        status = attendance.status
        sheet.append([employee_name, department, timestamp, status])

    # Set up the HTTP response with the Excel file
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=attendance_list.xlsx'

    # Save the workbook to the response
    wb.save(response)

    return response

@user_passes_test(lambda user: user.is_superuser, login_url='login')
def leave_list(request):
    leaves = Leaves.objects.all().order_by('-start_date')
    return render(request, 'admin/leave_list.html', {'leaves': leaves})

def update_leave_status(request, leave_id, status):
    leave = get_object_or_404(Leaves, leave_id=leave_id)
    
    if status in ["Approved", "Rejected", "Pending"]:
        leave.status = status
        leave.save()
        messages.success(request, f"Leave request {status.lower()} successfully!")
    
    return redirect('leave_list')

@user_passes_test(lambda user: user.is_superuser, login_url='login')
def apology_list(request):
    apologies = Apology.objects.all().order_by('-date')
    return render(request, 'admin/apology_list.html', {'apologies': apologies})


def update_apology_status(request, apology_id, status):
    apology = get_object_or_404(Apology, apology_id=apology_id)
    
    if status in ["Approved", "Rejected", "Pending"]:
        apology.status = status
        apology.save()
        messages.success(request, f"Apology request {status.lower()} successfully!")
    
    return redirect('apology_list')

import cv2
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from deepface import DeepFace
import base64
from io import BytesIO
from PIL import Image

# View to render the template and provide image capturing functionality
@user_passes_test(lambda user: user.is_superuser, login_url='login')
def capture_face(request):
    return render(request, 'attendance/recognition.html')

from deepface import DeepFace
from django.http import JsonResponse

import cv2
import numpy as np
import json
from django.http import JsonResponse
from deepface import DeepFace
from .models import Profile  # Import your Profile model
from datetime import date


from django.urls import reverse

@user_passes_test(lambda user: user.is_superuser, login_url='login')
def process_image(request):
    if request.method == "POST":
        image_file = request.FILES.get("image")

        if not image_file:
            return JsonResponse({"error": "No image uploaded"}, status=400)

        try:
            # Save the image temporarily
            image_path = "temp_image.jpg"
            with open(image_path, "wb") as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

            # Read the image using OpenCV
            image = cv2.imread(image_path)

            # Resize for faster processing (optional)
            image_resized = cv2.resize(image, (640, 480))

            # Extract faces using DeepFace
            detected_faces = DeepFace.extract_faces(image_resized, detector_backend="opencv")

            if detected_faces:
                face_data = detected_faces[0]
                face_array = face_data["face"]

                # Extract face encoding
                face_encoding = DeepFace.represent(face_array, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]

                # Compare with stored encodings
                recognized_users = []
                threshold = 1  # Tune this threshold

                for profile in Profile.objects.all():
                    if profile.face_encoding:
                        stored_encoding = np.array(json.loads(profile.face_encoding))
                        distance = np.linalg.norm(face_encoding - stored_encoding)
                        print("Distance:", distance)

                        if distance < threshold:
                            employee = Employee.objects.get(user=profile.user)

                            # Check if attendance is already marked today
                            if not Attendance.objects.filter(emp=employee, date=date.today()).exists():
                                Attendance.objects.create(
                                    emp=employee,
                                    date=date.today(),
                                    status="Present"
                                )
                                messages.success(request, "Attendance marked successfully. Thank You")
                                return JsonResponse({"redirect_url": reverse("capture_attendance")})

                            else:
                                messages.warning(request, "This face has already marked for today. Thank You")
                                return JsonResponse({"error": "Attendance already marked for today", "redirect_url": reverse("capture_attendance")})

                messages.warning(request, "No Match Found. Thank You")
                return JsonResponse({"error": "No Match Found. Thank You", "redirect_url": reverse("capture_attendance")})

            else:
                messages.warning(request, "No Face Detected. Try Again")
                return JsonResponse({"error": "No Face Detected, try again.", "redirect_url": reverse("capture_attendance")})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)


@user_passes_test(lambda user: user.is_superuser, login_url='login')
def mannual(request):
    # Get date from GET request, default to today if not provided
    selected_date = request.GET.get('date', timezone.now().date().strftime('%Y-%m-%d'))

    # Convert the selected date string to a date object
    try:
        selected_date = timezone.datetime.strptime(selected_date, "%Y-%m-%d").date()
    except ValueError:
        selected_date = timezone.now().date()  # Fallback to today's date if parsing fails

    active_employees = Employee.objects.filter(status="Active")  # Filter only active employees

    employee_status = []

    for employee in active_employees:
        attendance = Attendance.objects.filter(emp=employee, date=selected_date).first()
        status = attendance.status if attendance else "Need Action"  

        employee_status.append({
            "id": employee.employee_id,
            "name": f"{employee.user.first_name} {employee.user.last_name}",
            "department": employee.dep.name if employee.dep else "N/A",
            "role": employee.role,
            "status": status  
        })

    return render(request, 'attendance/mannual.html', {
        'employee_status': employee_status,
        'selected_date': selected_date.strftime('%Y-%m-%d')  # Pass selected date back to template
    })

def toggle_attendance(request, employee_id):
    today = timezone.now().date()  # Get today's date
    employee = get_object_or_404(Employee, pk=employee_id)

    # Try to get an existing attendance record for today
    attendance = Attendance.objects.filter(emp=employee, date=today).first()

    if attendance:  
        # If record exists, toggle the status
        if attendance.status == "Present":
             attendance.status = "Absent"
        elif attendance.status == "Absent":
             attendance.status = "Present"
        else: 
            attendance.status = "Present"
        attendance.save() 
    else:
        # If no record exists, create a new one and mark as present
        attendance = Attendance.objects.create(emp=employee, date=today, status="Present")

    messages.success(request, f"Attendance updated for {employee.user.first_name} {employee.user.last_name}.")
    return redirect('mannual')  # Redirect to attendance list

from django.contrib.humanize.templatetags.humanize import naturaltime
from django.utils.timesince import timesince
@login_required(login_url='login')
def employee_dashboard(request):
    today = now().date()  # Get today's date

    # Get the logged-in employee
    employee = Employee.objects.get(user=request.user)

    # Count attendance status for the logged-in user
    total_employees = Employee.objects.filter(status="Active").count()
    present = Attendance.objects.filter(emp=employee, date=today, status="Present").count()
    absent = Attendance.objects.filter(emp=employee, date=today, status="Absent").count()
    not_reported = total_employees-(present+absent)
    
    # Get the logged-in user's 5 most recent attendance records
    recent_attendance = Attendance.objects.filter(emp=employee).order_by('-date', '-attendance_id')[:4]

    # Get the logged-in user's 5 most recent apologies
    recent_apologies = Apology.objects.filter(emp=employee).order_by('-timestamp')[:2]

    for apology in recent_apologies:
        apology.time_since = timesince(apology.timestamp)  # Calculate time since submission

    context = {
        'present': present,
        'absent': absent,
        'total_employees':total_employees,
        'not_reported':not_reported,
        'recent_attendance': recent_attendance,
        'recent_apologies': recent_apologies,
    }

    return render(request, 'employee/dashboard.html', context)


@login_required
def my_attendance(request):
    user = request.user
    today = timezone.now().date()

    # Get the employee instance associated with the logged-in user
    employee = user.employee

    # Retrieve the logged-in user's attendance records
    attendance_records = Attendance.objects.filter(emp=employee).order_by('-date') # Last 5 records

    context = {
        'attendances': attendance_records,
    }

    return render(request, 'employee/my_attendance.html', context)
from .forms import LeaveApplicationForm
from .models import Leaves, Employee

@login_required
def leave_history(request):
    leaves = Leaves.objects.filter(emp=request.user.employee).order_by('-start_date')
    return render(request, 'employee/leave_history.html', {'leaves': leaves})

@login_required
def apply_leave(request):
    if request.method == "POST":
        form = LeaveApplicationForm(request.POST)
        if form.is_valid():
            leave = form.save(commit=False)
            leave.emp = Employee.objects.get(user=request.user)
            leave.save()
            messages.success(request, "Leave Submitted Successfuly")
            return redirect('leave_history')  # Redirect to success page
    else:
        form = LeaveApplicationForm()

    return render(request, 'employee/apply_leave.html', {'form': form})

@login_required
def delete_leave(request, leave_id):
    """Delete a leave request and redirect to the history page."""
    leave = get_object_or_404(Leaves, leave_id=leave_id, emp=request.user.employee)
    leave.delete()
    messages.success(request, "Leave request deleted successfully.")
    return redirect('leave_history')
@login_required
def edit_leave(request, leave_id):
    """Allow employees to edit their leave requests."""
    leave = get_object_or_404(Leaves, leave_id=leave_id, emp=request.user.employee)
    
    if request.method == "POST":
        form = LeaveApplicationForm(request.POST, instance=leave)
        if form.is_valid():
            form.save()
            messages.success(request, "Leave request updated successfully.")
            return redirect('leave_history')
    else:
        form = LeaveApplicationForm(instance=leave)

    return render(request, 'employee/apply_leave.html', {'form': form})

@login_required
def apology_history(request):
    apologies = Apology.objects.filter(emp=request.user.employee).order_by('-date')
    return render(request, 'employee/apology_history.html', {'apologies': apologies})

from .forms import ApologyForm

@login_required
def submit_apology(request):
    if request.method == 'POST':
        form = ApologyForm(request.POST)
        if form.is_valid():
            apology = form.save(commit=False)
            apology.emp = request.user.employee  # Assign the logged-in employee
            apology.save()
            messages.success(request, "Your apology has been submitted successfully.")
            return redirect('apology_history')  # Redirect to apology history page
    else:
        form = ApologyForm()
    
    return render(request, 'employee/submit_apology.html', {'form': form})

@login_required
def edit_apology(request, apology_id):
    """View to edit an existing apology."""
    apology = get_object_or_404(Apology, apology_id=apology_id, emp=request.user.employee)

    if request.method == 'POST':
        form = ApologyForm(request.POST, instance=apology)
        if form.is_valid():
            form.save()
            messages.success(request, "Apology updated successfully.")
            return redirect('apology_history')
    else:
        form = ApologyForm(instance=apology)

    return render(request, 'employee/submit_apology.html', {'form': form, 'apology': apology})

@login_required
def delete_apology(request, apology_id):
    """Delete an apology and redirect to the history page."""
    apology = get_object_or_404(Apology, apology_id=apology_id, emp=request.user.employee)
    apology.delete()
    messages.success(request, "Apology deleted successfully.")
    return redirect('apology_history')

