elif page == "Dataset":

    st.header("📂 Dataset Overview")

    if uploaded_file is None:
        st.warning("Please upload a CSV file.")
    else:

        st.subheader("Dataset Preview")

        st.dataframe(df)

        st.subheader("Dataset Statistics")

        st.write(df.describe())

        st.subheader("Missing Values")

        st.dataframe(df.isnull().sum())

        c1,c2,c3,c4=st.columns(4)

        c1.metric("Rows",df.shape[0])
        c2.metric("Columns",df.shape[1])
        c3.metric("Total Sales",f"{df['Sales'].sum():,.0f}")
        c4.metric("Average Sales",f"{df['Sales'].mean():,.2f}")