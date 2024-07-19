import pickle 
import streamlit as st

st.header("Job Recommendation System Using Machine Learning")

# Load the employment data
with open("artifact/employment_data.pkl", "rb") as file:
    jobs = pickle.load(file)

# Load the similarity data
with open("artifact/similarity_data.pkl", "rb") as file_sim:
    similarity = pickle.load(file_sim)

# Combine job titles and current designations
jobs_list_1 = jobs['jobTitle'].values 
jobs_list_2 = jobs['currentDesignation'].values
combined_jobs = list(set(jobs_list_1).union(set(jobs_list_2)))


# combining ids
applicant_id = jobs['applicantId'].values
jobid = jobs['jobId'].values
combined_ids = list(set(applicant_id).union(set(jobid)))


# creating function for applicant id and job id

def recommendation(title):
    # Check if the title exists in the DataFrame
    if title not in jobs['applicantId'].values and title not in jobs['jobId'].values:
        return []

    # Initialize lists to store distance indices
    distance_idx = []
    distance_idx_2 = []

   
    # Find the index of the given job title.
    if title in jobs['jobId'].values:
        idx_list = jobs[jobs['jobId'] == title].index
        idx = jobs.index.get_loc(idx_list[0])
        # Calculate similarity indices and sort them
        distance_idx = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)[1:5]

    # Find the index of the given current designation.
    if title in jobs['applicantId'].values:
        idx_2_list = jobs[jobs['applicantId'] == title].index
        idx_2 = jobs.index.get_loc(idx_2_list[0])
        # Calculate similarity indices and sort them
        distance_idx_2 = sorted(list(enumerate(similarity[idx_2])), key=lambda x: x[1], reverse=True)[1:5]

    # Retrieve the job titles for the top similar jobs, avoiding duplicates
    recommended_jobs = []
    seen = set()
    for i in distance_idx:
        job_title = jobs.iloc[i[0]].jobTitle
        if job_title not in seen:
            recommended_jobs.append(job_title)
            seen.add(job_title)
    for i in distance_idx_2:
        job_title = jobs.iloc[i[0]].currentDesignation
        if job_title not in seen:
            recommended_jobs.append(job_title)
            seen.add(job_title)

    return recommended_jobs

# Job selection box
selected_job = st.selectbox(
    "Type or select a job ID to get recommendation",
    combined_ids
)

# Display recommendations based on the selected job
if selected_job:
    recommended_jobs = recommendation(selected_job)
    st.write("Recommended Jobs:")
    for job in recommended_jobs:
        st.write(job)
