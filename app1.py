import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🛍️ Mall Customer Segmentation Dashboard")
st.markdown("K-Means Clustering on Age, Annual Income, and Spending Score")

@st.cache_data
def load_sample_data():
    csv_data = '''CustomerID,Gender,Age,Annual_Income,Spending_Score
1,Male,19,15,39
2,Male,21,15,81
3,Female,20,16,6
4,Female,23,16,77
5,Female,31,17,40
6,Female,22,17,76
7,Female,35,18,6
8,Female,23,18,94
9,Male,64,19,3
10,Female,30,19,72
11,Male,67,19,14
12,Female,35,19,99
13,Female,58,20,15
14,Male,24,20,77
15,Male,37,20,13
16,Male,22,20,79
17,Female,35,21,35
18,Male,20,21,66
19,Male,52,23,29
20,Female,35,23,98
21,Male,35,24,35
22,Male,25,24,73
23,Female,46,25,5
24,Male,31,25,73
25,Female,54,28,14
26,Male,29,28,82
27,Female,45,28,32
28,Male,35,28,61
29,Female,40,29,31
30,Female,23,29,87
31,Male,60,30,4
32,Male,21,30,73
33,Male,53,33,4
34,Female,18,33,92
35,Female,49,33,14
36,Female,21,33,81
37,Female,42,34,17
38,Female,30,34,73
39,Male,36,37,26
40,Female,20,37,75
41,Female,65,38,35
42,Male,24,38,92
43,Male,48,39,36
44,Female,31,39,61
45,Female,49,39,28
46,Female,24,39,65
47,Female,50,40,55
48,Female,27,40,47
49,Male,29,40,42
50,Female,31,40,42
51,Female,49,42,52
52,Male,33,42,60
53,Female,31,43,54
54,Male,59,43,60
55,Female,35,43,45
56,Female,37,43,41
57,Male,32,44,50
58,Male,19,44,46
59,Female,35,46,51
60,Female,47,46,46
61,Male,28,46,56
62,Male,32,46,55
63,Female,41,47,52
64,Female,36,47,59
65,Male,34,48,51
66,Female,32,48,59
67,Male,33,48,50
68,Female,38,48,48
69,Male,47,48,59
70,Male,40,48,47
71,Female,48,49,55
72,Male,32,49,42
73,Female,24,50,49
74,Male,40,50,56
75,Female,27,54,47
76,Male,48,54,54
77,Male,28,54,53
78,Male,30,54,48
79,Male,34,54,55
80,Male,23,54,52
81,Male,19,54,46
82,Female,31,54,51
83,Male,27,54,54
84,Female,60,54,49
85,Male,45,54,53
86,Female,32,54,60
87,Female,27,54,56
88,Male,38,54,47
89,Female,35,54,48
90,Male,40,54,56
91,Female,28,54,47
92,Male,24,54,53
93,Male,39,54,42
94,Female,24,54,51
95,Female,33,54,59
96,Male,30,54,56
97,Female,23,54,57
98,Female,40,54,56
99,Male,27,54,56
100,Female,24,54,58
101,Male,47,62,41
102,Male,27,62,55
103,Female,47,62,41
104,Female,26,62,55
105,Male,35,62,42
106,Male,32,62,42
107,Female,36,62,50
108,Female,48,62,46
109,Male,29,62,51
110,Male,31,62,46
111,Female,36,62,46
112,Male,30,62,42
113,Female,34,62,51
114,Male,49,63,46
115,Female,24,63,50
116,Male,50,63,43
117,Female,27,63,48
118,Female,29,63,52
119,Female,31,63,54
120,Male,43,63,42
121,Female,40,63,46
122,Male,45,63,52
123,Male,38,63,55
124,Male,19,64,46
125,Male,21,65,48
126,Male,43,65,52
127,Female,59,67,49
128,Female,21,67,49
129,Female,28,67,48
130,Male,36,67,47
131,Male,36,67,49
132,Female,38,67,49
133,Female,47,67,48
134,Female,32,67,47
135,Female,28,67,50
136,Female,29,67,46
137,Female,34,67,43
138,Female,36,67,48
139,Male,35,67,47
140,Female,31,69,50
141,Female,46,69,45
142,Female,29,70,46
143,Female,45,70,43
144,Male,19,71,59
145,Male,28,71,55
146,Female,19,71,56
147,Male,29,71,56
148,Female,31,71,54
149,Male,31,71,55
150,Female,29,72,54
151,Female,29,72,53
152,Female,46,73,55
153,Female,33,73,52
154,Male,30,73,56
155,Female,59,74,54
156,Male,26,74,55
157,Male,19,75,54
158,Female,31,75,59
159,Male,29,76,57
160,Male,29,76,54
161,Male,35,77,51
162,Male,38,77,57
163,Male,32,77,54
164,Male,46,78,54
165,Male,27,78,57
166,Female,28,78,60
167,Female,33,78,53
168,Female,38,78,57
169,Male,36,78,54
170,Male,34,78,55
171,Female,36,78,52
172,Male,47,78,59
173,Male,35,78,59
174,Female,30,78,57
175,Female,34,78,54
176,Female,35,78,57
177,Male,32,78,55
178,Male,34,78,54
179,Female,36,78,60
180,Female,31,78,61
181,Female,34,78,51
182,Male,35,78,51
183,Male,40,78,50
184,Female,30,78,57
185,Female,31,78,54
186,Male,38,78,56
187,Male,47,78,55
188,Female,41,78,52
189,Male,36,78,56
190,Male,34,78,55
191,Female,32,78,52
192,Male,32,78,54
193,Female,34,78,55
194,Female,36,78,52
195,Female,32,78,55
196,Female,34,78,54
197,Male,35,78,56
198,Female,28,78,52
199,Female,32,78,54
200,Male,35,78,55'''
    df = pd.read_csv(io.StringIO(csv_data))
    return df

df = load_sample_data()

st.sidebar.header("Clustering Controls")
k = st.sidebar.slider("Select number of clusters (k)", 2, 10, 5)
run_cluster = st.sidebar.button("Run K-Means Clustering")

tab1, tab2, tab3 = st.tabs(["Dataset", "Clustering", "Insights"])

with tab1:
    st.subheader("Raw Dataset")
    st.dataframe(df.head(10))
    st.write(f"Shape: {df.shape[0]} customers, {df.shape[1]} features")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Age", f"{df['Age'].mean():.1f}")
    with col2:
        st.metric("Average Income", f"${df['Annual_Income'].mean():.1f}k")
    with col3:
        st.metric("Average Spending", f"{df['Spending_Score'].mean():.1f}")

with tab2:
    if run_cluster:
        features = ['Age', 'Annual_Income', 'Spending_Score']
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        df['Segment'] = kmeans.fit_predict(X_scaled)

        segment_profile = df.groupby('Segment')[features].mean()
        names = []
        for i in range(k):
            income = segment_profile.loc[i, 'Annual_Income']
            spending = segment_profile.loc[i, 'Spending_Score']
            if income > 70 and spending > 60:
                names.append("Target Customers")
            elif income > 70 and spending < 40:
                names.append("Potential Customers")
            elif income < 40 and spending > 60:
                names.append("Careful Spenders")
            elif income < 40 and spending < 40:
                names.append("Standard Customers")
            else:
                names.append(f"Segment_{i}")

        df['Segment_Name'] = df['Segment'].map(dict(zip(range(k), names)))

        st.success(f"Clustering complete with k={k}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cluster Scatter Plot")
            fig, ax = plt.subplots(figsize=(8,6))
            sns.scatterplot(data=df, x='Annual_Income', y='Spending_Score', hue='Segment_Name',
                            palette='Set2', s=100, ax=ax)
            ax.set_title('Customer Segments')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col2:
            st.subheader("Segment Summary")
            summary = df.groupby('Segment_Name')[features].mean().round(1)
            summary['Count'] = df['Segment_Name'].value_counts()
            st.dataframe(summary)

        st.subheader("Segmented Data")
        st.dataframe(df)

        csv = df.to_csv(index=False)
        st.download_button("Download Segmented CSV", csv, "segmented_customers.csv", "text/csv")
    else:
        st.info("Click 'Run K-Means Clustering' in the sidebar to generate segments")

with tab3:
    st.subheader("Business Recommendations")
    st.markdown("""
    **Target Customers**: High income + high spending
    → VIP programs, premium products, loyalty rewards

    **Potential Customers**: High income + low spending
    → Targeted marketing, showcase product value

    **Careful Spenders**: Low income + high spending
    → Discounts, EMI options, bundle deals

    **Standard Customers**: Medium income + medium spending
    → Seasonal offers, maintain engagement
    """)
    st.subheader("Pre-generated Plots")
    st.image("download (3).png", caption="Elbow Method")
    st.image("download (2).png", caption="Income vs Spending")
    st.image("download (4).png", caption="separate Distribution")
    st.image("download (1).png", caption="Age Distribution")
    st.image("plot.png", caption="customer segment")