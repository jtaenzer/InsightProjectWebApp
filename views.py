from flask import render_template
from flask import request 
from InsightProjectWebApp import app

from joblib import load
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

save_path = "/home/ubuntu/InsightProjectWebApp/binaries/"

countvec = load(save_path + "count_vectorizer.joblib")
tfidf = load(save_path + "tfidf_transformer.joblib")
core_skills_dict = load(save_path + "core_skills_dict_key_str.joblib")
cluster_centroid_dict = load(save_path + "cluster_centroid_dict.joblib")

@app.route('/',methods=["GET","POST"]) #we are now using these methods to get user input
def home_page():
    return render_template('index.html')  # render a template

@app.route('/output')
def recommendation_output():
    # Pull input
    user_input = request.args.get('user_input')
    # Case if empty
    if user_input == "":
        return render_template("index.html",my_input=user_input,my_form_result="Empty")
    else:
        count_mat = countvec.transform([user_input])
        tfidf_mat = tfidf.transform(count_mat)
        user_vec = tfidf_mat.toarray()
        dist_list = list()
        label_list = list()
        for key in cluster_centroid_dict.keys():
            dist_list.append(np.linalg.norm(cluster_centroid_dict[key]["centroid"] - user_vec))
            label_list.append(cluster_centroid_dict[key]["label"])
        df_dist = pd.DataFrame(dist_list, columns=["distance"], index=label_list)
        df_dist_sorted = df_dist.sort_values(by="distance", axis=0)
        furthest = df_dist_sorted["distance"][-1]

        neighbours = list()
        neighbours_dist = list()
        for index, row in df_dist_sorted.iterrows():
            if len(neighbours) >= 3:
                break
            elif index not in neighbours:
                neighbours.append(index)
                neighbours_dist.append((1 - (row["distance"]/furthest))*100)

        title_0 = neighbours[0]
        title_1 = neighbours[1]
        title_2 = neighbours[2]
        title_3 = ""
        title_4 = ""

        comp_0 = "{:.2f}".format(neighbours_dist[0])
        comp_1 = "{:.2f}".format(neighbours_dist[1])
        comp_2 = "{:.2f}".format(neighbours_dist[2])
        comp_3 = ""
        comp_4 = ""

        user_input_list = user_input.split(", ")
        if len(user_input_list) < 2:
            user_input_list = user_input.split(",")

        skills_0 = list()
        missing_skills_0 = list()
        for skill in core_skills_dict[title_0]:
            if skill in user_input_list:
                skills_0.append(skill)
            else:
                missing_skills_0.append(skill)
        skills_0 = ", ".join(skills_0)
        missing_skills_0 = ", ".join(missing_skills_0)

        skills_1 = list()
        missing_skills_1 = list()
        for skill in core_skills_dict[title_1]:
            if skill in user_input_list:
                skills_1.append(skill)
            else:
                missing_skills_1.append(skill)
        skills_1 = ", ".join(skills_1)
        missing_skills_1 = ", ".join(missing_skills_1)

        skills_2 = list()
        missing_skills_2 = list()
        for skill in core_skills_dict[title_2]:
            if skill in user_input_list:
                skills_2.append(skill)
            else:
                missing_skills_2.append(skill)
        skills_2 = ", ".join(skills_2)
        missing_skills_2 = ", ".join(missing_skills_2)

        skills_3 = ""
        skills_4 = ""

        missing_skills_3 = ""
        missing_skills_4 = ""

        return render_template("index.html", my_input=user_input,
                               title_0=title_0, title_1=title_1, title_2=title_2, title_3=title_3, title_4=title_4,
                               comp_0=comp_0, comp_1=comp_1, comp_2=comp_2, comp_3=comp_3, comp_4=comp_4,
                               skills_0=skills_0, skills_1=skills_1, skills_2=skills_2, skills_3=skills_3, skills_4=skills_4,
                               missing_skills_0=missing_skills_0, missing_skills_1=missing_skills_1, missing_skills_2=missing_skills_2, missing_skills_3=missing_skills_3, missing_skills_4=missing_skills_4,
                               my_form_result="NotEmpty")
