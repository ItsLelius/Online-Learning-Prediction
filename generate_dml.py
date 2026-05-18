import pandas as pd
import numpy as np
import os

# ============================================================
# LOAD — already clean from Colab
# ============================================================
df = pd.read_csv('datasets/Cleaned_OnlineLearning.csv')

# Drop engineered features if present
engineered_cols = [
    'engagement_score', 'performance_score',
    'study_consistency', 'course_load_ratio',
    'activity_rate', 'cluster'
]
df = df.drop(columns=[c for c in engineered_cols if c in df.columns])
df['row_id'] = df.index + 1

print(f"Dataset loaded: {df.shape}")

# ============================================================
# HELPER
# ============================================================
def val(v):
    if v is None:
        return 'NULL'
    if isinstance(v, float) and np.isnan(v):
        return 'NULL'
    if str(v).strip() in ['nan', 'None', 'NaT', '']:
        return 'NULL'
    if isinstance(v, str):
        return f"'{v.strip().replace(chr(39), chr(39)+chr(39))}'"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(round(float(v), 2))
    return f"'{str(v)}'"

# ============================================================
# DISTRIBUTE INTO TABLES
# ============================================================

# COURSES — unique by course_name
courses = df.groupby('course_name').agg(
    course_category=('course_category', 'first'),
    difficulty_level=('difficulty_level', 'first'),
    total_modules=('total_modules', 'first'),
    total_quizzes=('total_quizzes', 'first'),
    course_duration_weeks=('course_duration_weeks', 'first')
).reset_index()
courses['course_id'] = courses.index + 1
print(f"Courses: {len(courses)} records")

# STUDENTS — unique by student_name
students = df.groupby('student_name').agg(
    age=('age', 'first'),
    gender=('gender', 'first'),
    region=('region', 'first'),
    internet_access_type=('internet_access_type', 'first'),
    education_level=('education_level', 'first'),
    employment_status=('employment_status', 'first'),
    device_type=('device_type', 'first'),
    weekly_study_hours=('weekly_study_hours', 'first'),
    learning_goal=('learning_goal', 'first'),
    prior_courses_completed=('prior_courses_completed', 'first')
).reset_index()
students['student_id'] = students.index + 1

def split_name(name):
    parts = str(name).strip().split(' ', 1)
    return parts[0], parts[1] if len(parts) > 1 else 'Unknown'

students['first_name'] = students['student_name'].apply(
    lambda x: split_name(x)[0]
)
students['last_name'] = students['student_name'].apply(
    lambda x: split_name(x)[1]
)
print(f"Students: {len(students)} records")

# MAP IDs
course_map = dict(zip(courses['course_name'], courses['course_id']))
student_map = dict(zip(students['student_name'], students['student_id']))
df['course_id'] = df['course_name'].map(course_map)
df['student_id'] = df['student_name'].map(student_map)

# ENROLLMENTS — one per student per course
enrollments = df.groupby(['student_id', 'course_id']).agg(
    enrollment_date=('enrollment_date', 'first'),
    completion_status=('completion_status', 'first')
).reset_index()
enrollments['enrollment_id'] = enrollments.index + 1
print(f"Enrollments: {len(enrollments)} records")

# ENGAGEMENT LOGS — one row per record
engagement_logs = df[[
    'row_id', 'student_id', 'course_id',
    'login_date', 'session_duration_minutes',
    'modules_accessed', 'videos_watched'
]].copy()
engagement_logs.columns = [
    'log_id', 'student_id', 'course_id',
    'login_date', 'session_duration_minutes',
    'modules_accessed', 'videos_watched'
]
print(f"Engagement Logs: {len(engagement_logs)} records")

# ASSESSMENTS — unique per course and type
assessments = df[['course_id', 'assessment_type']].drop_duplicates().reset_index(drop=True)
assessments['assessment_id'] = assessments.index + 1
print(f"Assessments: {len(assessments)} records")

# MAP assessment_id
assessment_map = {
    (r['course_id'], r['assessment_type']): r['assessment_id']
    for _, r in assessments.iterrows()
}
df['assessment_id'] = df.apply(
    lambda x: assessment_map.get((x['course_id'], x['assessment_type'])), axis=1
)

# STUDENT ASSESSMENT RESULTS — one row per record
results = df[['row_id', 'student_id', 'assessment_id', 'score_obtained']].copy()
results.columns = ['result_id', 'student_id', 'assessment_id', 'score_obtained']
print(f"Assessment Results: {len(results)} records")

# COURSE FEEDBACK — one per student per course
feedback = df.groupby(['student_id', 'course_id']).agg(
    satisfaction_rating=('satisfaction_rating', 'first')
).reset_index()
feedback['feedback_id'] = feedback.index + 1
print(f"Course Feedback: {len(feedback)} records")

# STUDENT COURSE SUMMARY — aggregated per student per course
summary = df.groupby(['student_id', 'course_id']).agg(
    total_logins=('login_date', 'count'),
    total_modules_accessed=('modules_accessed', 'sum'),
    total_videos_watched=('videos_watched', 'sum'),
    total_posts=('total_posts', 'sum'),
    avg_session_duration=('avg_session_duration', 'mean'),
    quiz_avg_score=('quiz_avg_score', 'mean'),
    assignment_avg_score=('assignment_avg_score', 'mean')
).reset_index()
summary['summary_id'] = summary.index + 1
print(f"Student Course Summary: {len(summary)} records")

# ============================================================
# GENERATE DML
# ============================================================
lines = []
lines.append("-- ============================================================")
lines.append("-- DML: Online Learning Prediction System")
lines.append("-- Source: Cleaned_OnlineLearning.csv")
lines.append("-- Note: Data already cleaned in Colab Notebook 1")
lines.append("-- This script only distributes columns into their tables")
lines.append("-- ============================================================")
lines.append("")
lines.append("USE online_learning_prediction;")
lines.append("SET FOREIGN_KEY_CHECKS = 0;")
lines.append("")

# STUDENTS
lines.append("-- INSERT: students")
for _, r in students.iterrows():
    lines.append(
        f"INSERT INTO students (student_id, first_name, last_name, age, "
        f"gender, region, internet_access_type, education_level, "
        f"employment_status, device_type, weekly_study_hours, "
        f"learning_goal, prior_courses_completed) VALUES ("
        f"{val(r['student_id'])}, {val(r['first_name'])}, {val(r['last_name'])}, "
        f"{val(r['age'])}, {val(r['gender'])}, {val(r['region'])}, "
        f"{val(r['internet_access_type'])}, {val(r['education_level'])}, "
        f"{val(r['employment_status'])}, {val(r['device_type'])}, "
        f"{val(r['weekly_study_hours'])}, {val(r['learning_goal'])}, "
        f"{val(r['prior_courses_completed'])});"
    )
lines.append("")

# COURSES
lines.append("-- INSERT: courses")
for _, r in courses.iterrows():
    lines.append(
        f"INSERT INTO courses (course_id, course_name, course_category, "
        f"difficulty_level, total_modules, total_quizzes, course_duration_weeks) "
        f"VALUES ({val(r['course_id'])}, {val(r['course_name'])}, "
        f"{val(r['course_category'])}, {val(r['difficulty_level'])}, "
        f"{val(r['total_modules'])}, {val(r['total_quizzes'])}, "
        f"{val(r['course_duration_weeks'])});"
    )
lines.append("")

# ENROLLMENTS
lines.append("-- INSERT: enrollments")
for _, r in enrollments.iterrows():
    lines.append(
        f"INSERT INTO enrollments (enrollment_id, student_id, course_id, "
        f"enrollment_date, completion_status) VALUES ("
        f"{val(r['enrollment_id'])}, {val(r['student_id'])}, "
        f"{val(r['course_id'])}, {val(r['enrollment_date'])}, "
        f"{val(r['completion_status'])});"
    )
lines.append("")

# ENGAGEMENT LOGS
lines.append("-- INSERT: engagement_logs")
for _, r in engagement_logs.iterrows():
    lines.append(
        f"INSERT INTO engagement_logs (log_id, student_id, course_id, "
        f"login_date, session_duration_minutes, modules_accessed, videos_watched) "
        f"VALUES ({val(r['log_id'])}, {val(r['student_id'])}, "
        f"{val(r['course_id'])}, {val(r['login_date'])}, "
        f"{val(r['session_duration_minutes'])}, {val(r['modules_accessed'])}, "
        f"{val(r['videos_watched'])});"
    )
lines.append("")

# ASSESSMENTS
lines.append("-- INSERT: assessments")
for _, r in assessments.iterrows():
    lines.append(
        f"INSERT INTO assessments (assessment_id, course_id, assessment_type) "
        f"VALUES ({val(r['assessment_id'])}, {val(r['course_id'])}, "
        f"{val(r['assessment_type'])});"
    )
lines.append("")

# STUDENT ASSESSMENT RESULTS
lines.append("-- INSERT: student_assessment_results")
for _, r in results.iterrows():
    lines.append(
        f"INSERT INTO student_assessment_results (result_id, student_id, "
        f"assessment_id, score_obtained) VALUES ({val(r['result_id'])}, "
        f"{val(r['student_id'])}, {val(r['assessment_id'])}, "
        f"{val(r['score_obtained'])});"
    )
lines.append("")

# COURSE FEEDBACK
lines.append("-- INSERT: course_feedback")
for _, r in feedback.iterrows():
    lines.append(
        f"INSERT INTO course_feedback (feedback_id, student_id, course_id, "
        f"satisfaction_rating) VALUES ({val(r['feedback_id'])}, "
        f"{val(r['student_id'])}, {val(r['course_id'])}, "
        f"{val(r['satisfaction_rating'])});"
    )
lines.append("")

# STUDENT COURSE SUMMARY
lines.append("-- INSERT: student_course_summary")
for _, r in summary.iterrows():
    lines.append(
        f"INSERT INTO student_course_summary (summary_id, student_id, course_id, "
        f"total_logins, total_modules_accessed, total_videos_watched, total_posts, "
        f"avg_session_duration, quiz_avg_score, assignment_avg_score) VALUES ("
        f"{val(r['summary_id'])}, {val(r['student_id'])}, {val(r['course_id'])}, "
        f"{val(r['total_logins'])}, {val(r['total_modules_accessed'])}, "
        f"{val(r['total_videos_watched'])}, {val(r['total_posts'])}, "
        f"{val(r['avg_session_duration'])}, {val(r['quiz_avg_score'])}, "
        f"{val(r['assignment_avg_score'])});"
    )
lines.append("")
lines.append("SET FOREIGN_KEY_CHECKS = 1;")
lines.append("-- DML Complete")

# ============================================================
# SAVE
# ============================================================
os.makedirs('sql', exist_ok=True)
output_path = 'sql/DML_online_learning_prediction.sql'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"\nDML generated successfully.")
print(f"Saved to: {output_path}")