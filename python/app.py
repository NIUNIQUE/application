import streamlit as st
import requests
from bs4 import BeautifulSoup
import jieba
from collections import Counter
import re
import os
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from altair import Chart, X, Y

# 用于保存临时文件
TEMP_DIR = './temp'
os.makedirs(TEMP_DIR, exist_ok=True)

def fetch_text_from_url(url):
    """从URL获取文本内容"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.body.get_text(strip=True, separator='\n') if soup.body else ''

def preprocess_text(text):
    """文本预处理"""
    text = re.sub(r'[<.*?>]|[^\u4e00-\u9fffA-Za-z]', ' ', text)  # 去除HTML标签，保留汉字和英文字符
    return re.sub(r'\s+', ' ', text).strip()

def segment_and_count(text, stopwords):
    """分词并统计词频"""
    words = [word for word in jieba.cut(text) if word.strip() and word not in stopwords]
    return Counter(words)

def generate_word_cloud(word_counts):
    """生成词云图"""
    word_cloud = WordCloud(
        font_path="./simfang.ttf",  # 指定中文字体
        width=800,
        height=400,
        background_color="white",
        max_words=200
    ).generate_from_frequencies(word_counts)
    buf = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf

def create_plotly_chart(chart_type, word_counts, top_n=20):
    """使用Plotly创建图表"""
    df = pd.DataFrame(word_counts.most_common(top_n), columns=["词语", "频率"])
    chart_mapping = {
        "垂直条形图": px.bar(df, x="词语", y="频率", title="垂直条形图"),
        "水平条形图": px.bar(df, x="频率", y="词语", orientation='h', title="水平条形图"),
        "饼图": px.pie(df, values="频率", names="词语", title="饼图"),
        "折线图": px.line(df, x="词语", y="频率", title="折线图"),
        "散点图": px.scatter(df, x="词语", y="频率", size="频率", title="散点图")
    }
    return chart_mapping.get(chart_type)

def create_plotly_radar_chart(word_counts, top_n=10):
    """使用Plotly绘制雷达图"""
    top_words = word_counts.most_common(top_n)
    if not top_words:
        return None
    words, counts = zip(*top_words)
    fig = px.line_polar(r=list(counts), theta=list(words), line_close=True, title="雷达图")
    fig.update_traces(fill='toself', marker=dict(size=5))
    return fig

def create_altair_chart(word_counts, top_n=20):
    """使用Altair创建面积图"""
    df = pd.DataFrame(word_counts.most_common(top_n), columns=["词语", "频率"])
    return Chart(df).mark_area(opacity=0.5).encode(
        X("词语:N", title="词语"),
        Y("频率:Q", title="频率")
    ).properties(title="面积图", width=600, height=400)

def render_word_cloud(word_counts):
    """渲染词云图到Streamlit"""
    buf = generate_word_cloud(word_counts)
    img = base64.b64encode(buf.read()).decode()
    buf.close()
    st.markdown(f"![词云图](data:image/png;base64,{img})")

def main():
    st.title("文章URL文本分析工具")

    url = st.text_input("请输入文章的URL:")
    chart_type = st.sidebar.selectbox(
        "选择图表类型",
        ["词云", "垂直条形图", "水平条形图", "饼图", "折线图", "散点图", "雷达图", "面积图"],
    )

    if not url:
        st.warning("请输入有效的URL以进行分析！")
        return

    if st.button("提交"):
        with st.spinner('正在处理...'):
            text = fetch_text_from_url(url)
            processed_text = preprocess_text(text)

            # 加载停用词表
            with open('stopwords.txt', 'r', encoding='utf-8') as f:
                stopwords = set(line.strip() for line in f)

            word_counts = segment_and_count(processed_text, stopwords)
            st.write("词频最高的20个词：")
            for word, count in word_counts.most_common(20):
                st.write(f"{word}: {count}")

            # 根据选择的图表类型生成对应图表
            if chart_type == "词云":
                render_word_cloud(word_counts)
            elif chart_type == "面积图":
                chart = create_altair_chart(word_counts)
                st.altair_chart(chart, use_container_width=True)
            elif chart_type == "雷达图":
                chart = create_plotly_radar_chart(word_counts)
                st.plotly_chart(chart, use_container_width=True)
            else:
                chart = create_plotly_chart(chart_type, word_counts)
                st.plotly_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()
