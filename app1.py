# ----------------------------------------------------------------------
# 毕业设计：基于Python的IT行业招聘数据可视化分析系统

# ----------------------------------------------------------------------

# --- 核心库导入 ---
# streamlit 实现用户交互界面
import streamlit as st
# pandas 数据加载、筛选和聚合
import pandas as pd
# plotly 绘制交互性图表
import plotly.express as px
#os 检查文件是否存在、路径是否正确
import os
# wordcloud&jieba 文本挖掘&生成词云
import jieba
from wordcloud import WordCloud
# matplotlib 为词云提供画布
import matplotlib.pyplot as plt
# pydeck 绘制岗位热力图
import pydeck as pdk

# ======================================================================
#   (A) 页面基础设置、全局常量、核心函数定义
# ======================================================================
st.set_page_config(page_title="IT行业招聘数据分析系统", page_icon="💼", layout="wide")

# --- 全局常量 ---
#  定义“城市等级”、“工作经验”、“经验等级”、“学历”和公司规模的逻辑顺序与规则。
CITY_TIER_MAP = {
    '北京': '一线城市', '上海': '一线城市', '广州': '一线城市', '深圳': '一线城市',
    '成都': '新一线城市', '杭州': '新一线城市', '重庆': '新一线城市', '武汉': '新一线城市',
    '苏州': '新一线城市', '西安': '新一线城市', '南京': '新一线城市', '长沙': '新一线城市',
    '天津': '新一线城市', '郑州': '新一线城市', '东莞': '新一线城市', '青岛': '新一线城市',
    '合肥': '新一线城市', '佛山': '新一线城市', '宁波': '新一线城市'
}
EXPERIENCE_ORDER = ['无经验/应届生', '1年以内', '1-3年', '3-5年', '5-10年', '10年以上']
EXPERIENCE_LABEL_MAPPING = {0: '无经验/应届生', 1: '1年以内', 2: '1-3年', 3: '3-5年', 4: '5-10年', 5: '10年以上',
                            6: '不详'}
EDUCATION_ORDER = ['中专/中技', '高中', '大专', '本科', '硕士', '博士']
COMPANY_SIZE_ORDER = ['少于15人', '15-50人', '50-150人', '150-500人', '500-1000人', '1000-5000人', '5000-10000人',
                      '10000人以上', '不详']


#如此样式 @st.cache_data是为了避免了重复的网络请求、对相同数据集的重复Pandas计算、对相同文本的重复jieba分词和WordCloud渲染。

# --- 数据加载与准备函数 健壮性检查
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path): st.error(f"错误: 未找到数据文件 '{file_path}'。"); return None
    # 使用Pandas的read_csv函数，将CSV文件的内容读取到一个DataFrame对象中。
    df = pd.read_csv(file_path)
    # 根据我们之前定义的CITY_TIER_MAP字典替换城市相对应的城市等级如北京：一线城市
    df['城市等级'] = df['检索城市'].map(CITY_TIER_MAP).fillna('其他城市')
    return df


@st.cache_data#将数据存储与缓存
def prepare_experience_data(_df, mode='overall', cache_key=None):
    """
    一个通用的函数，用于准备“经验回报率”分析所需的数据。
    通过 mode 参数，可以为不同群体（总体、本科、核心本科）生成数据。
    :param _df: 输入的、经过筛选的 DataFrame。
    :param mode: 字符串，分析模式。可选值为 'overall', 'bachelor', 'core'。
    :return: 两个DataFrame，分别用于绘制平均值和中位数图表。
    """
    if mode == 'bachelor':
        source_df = _df[_df['学历'] == '本科'].copy()
    elif mode == 'core':
        bachelor_df = _df[_df['学历'] == '本科'].copy()
        # 只保留薪资在Q1和Q3之间的数据，即中间50%的核心数据。
        def remove_outliers(group):
            q1 = group['月薪'].quantile(0.25)
            q3 = group['月薪'].quantile(0.75)
            return group[(group['月薪'] >= q1) & (group['月薪'] <= q3)]
        # 为防止后续groupby出错，先丢弃'经验等级'为空的行，按'经验等级'分组，并对每个组应用remove_outliers函数。
        bachelor_df_no_na = bachelor_df.dropna(subset=['经验等级'])
        source_df = bachelor_df_no_na.groupby('经验等级').apply(remove_outliers).reset_index(drop=True)
    else:  # 默认为 'overall'
        source_df = _df.copy()
    # 无论何种模式，最终的分析都只关注这两个最具代表性的城市等级。
    analysis_df = source_df[source_df['城市等级'].isin(['一线城市', '新一线城市'])]
    ordered_exp_dtype = pd.CategoricalDtype(categories=EXPERIENCE_ORDER, ordered=True)

    # 计算平均值
    mean_df = analysis_df.groupby(['城市等级', '经验等级'])['月薪'].mean().round(0).reset_index()
    mean_df['经验标签'] = mean_df['经验等级'].map(EXPERIENCE_LABEL_MAPPING)
    mean_df['经验标签'] = mean_df['经验标签'].astype(ordered_exp_dtype)
    mean_df = mean_df.sort_values(by=['城市等级', '经验标签'])

    # 计算中位数
    median_df = analysis_df.groupby(['城市等级', '经验等级'])['月薪'].median().round(0).reset_index()
    median_df['经验标签'] = median_df['经验等级'].map(EXPERIENCE_LABEL_MAPPING)
    median_df['经验标签'] = median_df['经验标签'].astype(ordered_exp_dtype)
    median_df = median_df.sort_values(by=['城市等级', '经验标签'])

    return mean_df, median_df


 #准备“学历价值分析”所需的数据。
@st.cache_data
def prepare_education_data(_df, cache_key=None):
    # 只保留主流的学历层次
    edu_to_analyze = ['中专/中技', '高中', '大专', '本科', '硕士', '博士']
    # 从传入的DataFrame(_df)中，只筛选出'学历'列的值在我们上面定义的列表中的行
    analysis_df = _df[_df['学历'].isin(edu_to_analyze)]
    #  .groupby('学历'): 按“学历”将数据分成不同的组（大专组、本科组、硕士组...）。
    # ['月薪']: 在每个组内，我们只关注“月薪”这一列。
    # .median(): 计算每个组月薪的“中位数”，这是最能代表典型水平的指标。
    # .round(0): 将计算出的薪资中位数四舍五入到整数，让数字更整洁。
    # .reset_index(): 将groupby后的特殊格式，转换回一个标准的DataFrame
    edu_salary_median = analysis_df.groupby('学历')['月薪'].median().round(0).reset_index()
    ordered_edu_dtype = pd.CategoricalDtype(categories=EDUCATION_ORDER, ordered=True)
    edu_salary_median['学历'] = edu_salary_median['学历'].astype(ordered_edu_dtype)
    edu_salary_median = edu_salary_median.sort_values(by='学历')
    return edu_salary_median


# 准备“岗位类别分析”所需的数据。
@st.cache_data
def prepare_category_data(_df, cache_key=None):
    # .groupby('检索二级职位类别'): 按二级职位类别（如'后端开发', '人工智能'）分组。
    # .size(): 计算每个组里有多少行，也就是该类别的岗位数量。
    # .reset_index(name='岗位数量'): 将结果转为DataFrame，并给数量列命名为'岗位数量'。
    hot_jobs = _df.groupby('检索二级职位类别').size().reset_index(name='岗位数量')
    # .sort_values(by='岗位数量', ascending=False): 按岗位数量从高到低排序。
    # .head(15): 只取排序后的前15行。
    top_15_hot = hot_jobs.sort_values(by='岗位数量', ascending=False).head(15)
    high_salary_jobs = _df.groupby('检索二级职位类别')['月薪'].agg(['median', 'size']).reset_index()
    high_salary_jobs.columns = ['检索二级职位类别', '中位数月薪', '岗位数量']
    # 为了数据的准确性只保留那些岗位数量大于等于50的类别，确保薪资中位数具有统计学意义。
    significant_jobs = high_salary_jobs[high_salary_jobs['岗位数量'] >= 50]
    # 在具有统计意义的岗位中，按'中位数月薪'从高到低排序，并取前15名。
    top_15_high = significant_jobs.sort_values(by='中位数月薪', ascending=False).head(15)
    return top_15_hot, top_15_high


 # 准备“企业画像分析”所需的数据。
def prepare_company_data(_df):

    # 公司规模分析
    size_analysis = _df.groupby('公司规模标签')['月薪'].agg(['median', 'size']).round(0)
    size_analysis.columns = ['中位数月薪', '岗位数量']

    # 因为在数据清洗时，已经将'公司规模标签'定义为有序分类类型，
    # 所以 sort_index() 会自动按照期望的顺序（从小到大）进行排序。
    size_analysis = size_analysis.sort_index()

    # 公司类型分析
    type_analysis = _df.groupby('公司类型标签')['月薪'].agg(['median', 'size']).round(0)
    type_analysis.columns = ['中位数月薪', '岗位数量']
    # 为了图表美观，过滤掉样本量过小和“不详”的类别
    type_analysis = type_analysis[(type_analysis['岗位数量'] >= 50) & (type_analysis.index != '不详')]
    type_analysis = type_analysis.sort_values(by='中位数月薪', ascending=False)

    return size_analysis, type_analysis


# 词云模块准备数据
@st.cache_data
def generate_wordcloud_image(_df, column_name, use_stopwords=False, cache_key=None):
    # 检查要分析的列名是否存在于传入的DataFrame中，防止KeyError
    if column_name not in _df.columns: return None
    #.dropna(): 丢弃所有空值行，防止错误。
    #.join(...): 将指定列的所有文本，用空格连接成一个巨大的字符串
    text = " ".join(item for item in _df[column_name].dropna())
    # 如果拼接后是空字符串，则直接返回，避免后续计算
    if not text.strip(): return None
    # 调用jieba库的cut方法，对长文本进行精准的中文分词，得到一个词语的生成器。
    word_list = jieba.cut(text)

    if use_stopwords:
        stopwords = {...}  #停用词列表
        # 列表推导式：遍历分词后的所有词语，只保留那些长度大于1并且不在停用词列表里的词。
        filtered_words = [word for word in word_list if len(word) > 1 and word not in stopwords]
    else:
        # 画福利词云时，则只执行过滤单字的操作，不需要进行去除噪音数据。
        filtered_words = [word for word in word_list if len(word) > 1]

    if not filtered_words: return None

    # 检查字体文件
    font_path = 'simhei.ttf'
    if not os.path.exists(font_path):
        st.error(f"错误: 未找到中文字体 '{font_path}'。")
        return None
    # WordCloud(...): 初始化一个词云对象，并配置各种参数（字体、背景色、尺寸、最大词数等）。
    # .generate(" ".join(filtered_words))接收净化后的词语列表，在内部完成词频计算、根据词频确定大小、布局渲染所有工作。
    wordcloud = WordCloud(
        font_path=font_path, background_color="white", width=1000, height=500, max_words=100
    ).generate(" ".join(filtered_words))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig


# github上传数据
@st.cache_data
def load_data_from_url(url):
    """从URL加载并预处理数据"""
    try:
        df = pd.read_csv(url, compression='zip')

        #  去除 "_x000D_"字符
        # 对所有文本类型的列，进行一次性替换
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.replace('_x000D_', '', regex=False)

        df['城市等级'] = df['检索城市'].map(CITY_TIER_MAP).fillna('其他城市')
        return df
    except Exception as e:
        st.error(f"从URL加载数据时出错: {e}")
        return None


# ======================================================================
#   (B) UI与主逻辑
# ======================================================================

st.title("📊 IT行业招聘数据可视化分析系统")

# 定义从GitHub获取的原始数据文件URL (Raw URL)

DATA_URL = "https://github.com/ling-wei-yu/it-job/releases/download/V1.0/it_data_cleaned_v6_final.zip"

# 调用函数从URL加载数据
df = load_data_from_url(DATA_URL)

# 检查数据是否加载成功
if df is None:
    st.warning("数据加载失败，应用无法继续。请检查URL是否正确或网络连接。")
    st.stop()

# 侧边栏筛选器
st.sidebar.header("🔬 全局筛选器")
# 从DataFrame的'检索城市'列中，获取所有不重复的城市名，且让城市进行排序变得更加整洁。（学历、工作经验如上）
selected_cities = st.sidebar.multiselect('选择城市 (可多选)', options=sorted(df['检索城市'].unique()), default=[])
selected_education = st.sidebar.multiselect('选择学历 (可多选)', options=sorted(df['学历'].unique()), default=[])
selected_experience = st.sidebar.multiselect('选择工作经验 (可多选)', options=sorted(df['经验标签'].unique()),
                                             default=[])
# 从DataFrame (df) 复制一份命名为 df_filtered。后续所有的筛选操作，都只在这个副本上进行，确保了原始数据 df 永远不受污染，可以在下一次筛选时被重复使用。
df_filtered = df.copy()
if selected_cities: df_filtered = df_filtered[df_filtered['检索城市'].isin(selected_cities)]
if selected_education: df_filtered = df_filtered[df_filtered['学历'].isin(selected_education)]
if selected_experience: df_filtered = df_filtered[df_filtered['经验标签'].isin(selected_experience)]

st.sidebar.write("---")
# 这个数字会随着用户的每一次筛选而实时变化，提供了极佳的交互反馈
st.sidebar.metric(label="符合条件的岗位总数", value=f"{len(df_filtered)}")
st.sidebar.info("组合使用筛选器，所有图表都将实时更新。")

st.write("---")

# ======================================================================
#   (C) “双模态”智能渲染逻辑
# ======================================================================
st.write("---")

# --- 1. 决定用于展示的数据 ---
is_filtered = bool(selected_cities or selected_education or selected_experience)

if is_filtered:
    df_display = df_filtered
else:
    df_display = df

# --- 2. "无数据"的防御性检查 ---
if df_display.empty:
    st.warning("在此筛选条件下，没有找到足够的数据用于分析。请尝试放宽筛选条件。")

# --- 3. 根据模式，渲染不同的UI ---
elif not is_filtered:
    # ---------------------------------
    #   模式一：“宏观概览” (当无筛选时)
    # ---------------------------------
    st.info(
        "ℹ️ **您正在查看【宏观概览】。** 这里展示的是基于**全部数据**的总体趋势。您可以使用左侧的筛选器，对特定群体进行深度下钻分析。")

    # --- 模块一：经验回报率 ---（折线图）
    st.header("1. 经验回报率分析：多群体对比")
    # 使用Streamlit的st.tabs功能，创建了三个可以切换的标签页
    tab1, tab2, tab3 = st.tabs(["📈 总体趋势分析", "🎓 本科生专属分析", "🎯 核心本科生分析"])
    with tab1:
        st.subheader("全量数据：平均值 vs. 中位数")
        # 传入完整的数据集进行总体分析。col1/col2并排展示“平均值”和“中位数”两张图表。
        mean_data, median_data = prepare_experience_data(df_display, mode='overall')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("###### 平均薪资增长曲线")
            fig_mean = px.line(mean_data, x='经验标签', y='月薪', color='城市等级', markers=True,
                               title="工作经验对“平均薪资”的增长回报率")
            fig_mean.update_layout(title_x=0.5, title_font_size=16);
            st.plotly_chart(fig_mean, use_container_width=True)
        with col2:
            st.markdown("###### 中位数薪资增长曲线")
            fig_median = px.line(median_data, x='经验标签', y='月薪', color='城市等级', markers=True,
                                 title="工作经验对“中位数薪资”的增长回报率")
            fig_median.update_layout(title_x=0.5, title_font_size=16);
            st.plotly_chart(fig_median, use_container_width=True)

            # 两级结论模式是为了保证网页整洁的同时对于数据图表进行深度的探讨.(宏观模式下各模块二级结论作用类似)
            # 第一级：核心概要
        st.markdown(
            "核心结论：无论是平均值还是中位数，一线城市的起薪与经验回报率均高于新一线城市。同时，平均值显著高于中位数，揭示了薪酬的右偏态分布。")

        # 第二级：可展开的详细解读
        with st.expander("点击查看详细解读 👇"):
            st.markdown("""
                *   **城市差异:** 对比两条“中位数”增长曲线，一线城市的应届生起薪（约10,000元）比新一线城市（约8,500元）高出约17%。随着经验增长至“5-10年”，这一薪酬差距被进一步拉大。
                *   **统计学洞察:** “平均薪资”曲线全程显著高于“中位数薪资”曲线，这直观地证明了IT行业薪酬的“右偏态”分布特征。这意味着，少数薪酬极高的“明星”岗位，对整体的平均水平有强烈的拉升效应。因此，对于普通求职者，中位数是更具参考价值的基准。
                *   **提出疑问:** 即使是中位数，应届生的起薪也接近万元。这个数字是否仍然受到了少数“天才”毕业生的影响？为了回答这个问题，我们将在下一个标签页中，聚焦于占比最大的“本科生”群体。
                """)
    with tab2:
        st.subheader("本科生群体：更具代表性的薪酬轨迹")
        # 调用总体数据分析函数_median_data，使用下划线_分析本科生薪资回报率的中位数,px:绘制函数.
        _, bachelor_median_data = prepare_experience_data(df_display, mode='bachelor')
        fig_bachelor = px.line(bachelor_median_data, x='经验标签', y='月薪', color='城市等级', markers=True,
                               title="本科生专属：工作经验对“中位数薪资”的增长回报率")
        fig_bachelor.update_layout(title_x=0.5, title_font_size=16);
        st.plotly_chart(fig_bachelor, use_container_width=True)
        st.info("""
                **分析结论:**
                *   **聚焦主体:** 此图仅筛选“本科学历”的岗位进行分析，排除了其他学历的干扰，其结论更贴近普通本科毕业生的职业发展轨迹。
                *   **提出进一步疑问:** 我们观察到，本科应届生的中位数起薪（约8-9k）相较于总体市场的平均薪资（过万）更为贴近现实。**然而，这个中位数是否仍然受到了少数“天才本科生”高薪岗位的离群值影响?**
                """)
    with tab3:
        st.subheader("核心本科生群体：剔除离群值后的精细化分析")
        # 去掉薪资最高和最低的25%，只分析中间50%的市场主要力量、最保守也最具参考价值的薪资增长曲线，得出结论。
        _, core_bachelor_data = prepare_experience_data(df_display, mode='core')
        fig_core = px.line(core_bachelor_data, x='经验标签', y='月薪', color='城市等级', markers=True,
                           title="核心本科生(薪资25%-75%)：对“中位数薪资”的回报率")
        fig_core.update_layout(title_x=0.5, title_font_size=16);
        st.plotly_chart(fig_core, use_container_width=True)
        st.info(
            """
            **最终分析:**
            *   **最终基准:** 此图在“本科生”群体基础上，进一步剔除了每个经验等级中薪资最高和最低的25%的离群值，旨在反映市场中最“**普通大多数**”的薪酬变化趋势。这为普通本科生提供了一个**最保守、也最具参考价值**的薪酬期望基准。
            *   **有趣的发现:** 在“无经验/应届生”阶段，核心群体的中位数薪资甚至略高于全体本科生，这揭示了在应届生市场，存在大量低薪岗位，其对中位数的“下拉”效应不容忽视。
            """
        )

    st.write("---")

    # --- 模块二：学历价值 ---（带数值标签的条形图）
    st.header("2. 学历价值分析")
    # 调用我们之前已经定义好的使用@st.cache_data缓存的prepare_education_data数据准备函数
    edu_data = prepare_education_data(df_display)
    fig_edu = px.bar(edu_data, x='学历', y='月薪', text='月薪', color='学历', template='plotly_white',
                     title="不同学历层次的IT岗位薪酬中位数对比")
    # 为图表美观，隐藏图例、x/y轴，将数据显示在数据柱上将图表 fig_edu挂到Streamlit的网页上。
    # 且动态调整Y轴范围获取当前数据中“月薪”的最大值，并乘以1.15（为图表美观，留出15%的空白空间，确保不会顶到模块空间顶部）。
    fig_edu.update_layout(title_x=0.5, xaxis_title=None, yaxis_visible=False, showlegend=False,
                          yaxis_range=[0, edu_data['月薪'].max() * 1.15 if not edu_data.empty else 10000])
    fig_edu.update_traces(texttemplate='%{y:,.0f} 元', textposition='outside');
    st.plotly_chart(fig_edu, use_container_width=True)
    # 第一级：核心概要
    st.markdown("核心结论：学历水平与薪酬中位数存在清晰的阶梯效应，研究生及以上教育带来的薪酬跃升尤为显著。")

    # 第二级：可展开的详细解读
    with st.expander("点击查看详细解读 👇"):
        st.markdown("""
        *   **量化阶梯:** 从“大专”的9,000元，到“博士”的30,000元，图表清晰地展示了高等教育在IT职场的直接金钱回报。
        *   **“本硕”分水岭:** ‘本科’学历的中位数薪酬（15,000元）是进入IT行业主流岗位的“基础门票”。而从“本科”到“硕士”（17,500元），再到“博士”（30,000元），薪酬实现了两次显著的、非线性的跃升，这凸显了研究生教育对于冲击高薪岗位的决定性作用。
        *   **一个有趣的发现:** 在我们的数据中，“本科”与“本科及以上”的薪酬中位数完全相同，这可能表明在薪酬主体上，这两个要求在市场上的定位高度重合。
        """)

    st.write("---")

    # --- 模块三：岗位类别 ---（两个并排的水平条形图）
    st.header("3. 市场热点分析：热门 vs. 高薪岗位类别")
    # 调用上面准备好的数据分析函数，分成col3/col4图表(条形图)
    hot_data, high_salary_data = prepare_category_data(df_display)
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("热门岗位 TOP 15 (按需求量)")

        # 使用“链式调用”，完成所有配置。将创建图表 (px.bar) 和后续所有美化、配置的操作 (.update_layout, .update_traces)存放在同一个代码块中，
        #这样使得代码的可读性上升，增加代码的可组合性。
        fig_hot = (px.bar(
            hot_data, x='岗位数量', y='检索二级职位类别', text='岗位数量',
            orientation='h', template='plotly_white', title='IT行业热门岗位 TOP 15',
            labels={'检索二级职位类别': '职位类别'}  # 重命名Tooltip
        ).update_layout(
            title_x=0.5, xaxis_title='岗位数量 (个)', yaxis_title=None,
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=150)  # 增加左边距
        ).update_traces(
            textposition='outside'
        ))
        st.plotly_chart(fig_hot, use_container_width=True)

    with col4:
        st.subheader("高薪岗位 TOP 15 (按中位数月薪)")

        # 使用“链式调用”，作用如上。
        fig_high = (px.bar(
            high_salary_data, x='中位数月薪', y='检索二级职位类别', text='中位数月薪',
            orientation='h', template='plotly_white', title='IT行业高薪岗位 TOP 15',
            labels={'检索二级职位类别': '职位类别'}
        ).update_layout(
            title_x=0.5, xaxis_title='中位数月薪 (元)', yaxis_title="职位类别",
            yaxis={'categoryorder': 'total ascending'}
        ).update_traces(
            texttemplate='%{x:,.0f} 元',
            textposition='outside'
        ))
        st.plotly_chart(fig_high, use_container_width=True)

    # 第一级：核心概要
    st.markdown(
        "核心结论：“后端开发”是市场需求最广的岗位，而“人工智能”则是薪酬回报最高的领域，揭示了“热门”与“高薪”的差异。")

    # 第二级：可展开的详细解读
    with st.expander("点击查看详细解读 👇"):
        st.markdown("""
        *   **需求端分析 (热门榜):** `后端开发`、`技术管理`、`前端/移动开发` 占据了需求量的前三甲，是构成IT行业用人需求的“基本盘”。值得注意的是，`人工智能` 的需求量已超过 `测试`，位列第四，显示了其强劲的发展势头。
        *   **价值端分析 (高薪榜):** `人工智能` 以超过20,000元的中位数月薪，无可争议地成为“薪酬之王”。紧随其后的是 `高端技术职位` 和 `技术管理`，这清晰地指明了“精深技术”与“走向管理”是IT从业者的两条黄金晋升路线。
        *   **“金字塔”结构洞察:** `测试`、`运维/技术支持` 等岗位在“热门榜”上名列前茅，但在“高薪榜”上却不见踪影。这揭示了IT行业的“金字塔”结构：塔基是保证行业运转、需求量大但薪酬普遍的岗位；塔尖则是推动行业创新、技术壁垒高且薪酬丰厚的岗位。
        """)

    # 模块四企业画像分析（两个并排的条形图）
    st.header("4. 企业画像分析：规模与性质的对比")
    # 调用核心函数定义缓存的数据
    size_data, type_data = prepare_company_data(df_filtered)

    col5, col6 = st.columns(2)
    # 公司规模图。x=size_data.index：因为.groupby('公司规模标签') 创建了公司规模标签，所以直接调用结果DataFrame的索引 (index)为X轴。
    with col5:
        st.subheader("不同公司规模的薪酬对比")
        fig_size = px.bar(size_data, x=size_data.index, y='中位数月薪', text='中位数月薪',
                          title='公司规模 vs. 薪酬中位数')
        fig_size.update_layout(title_x=0.5, xaxis_title=None)
        fig_size.update_traces(texttemplate='%{y:,.0f} 元', textposition='outside')
        # size_data['中位数月薪'].max() * 1.15]：为图表美观，动态设置Y轴。
        if not size_data.empty:
            fig_size.update_yaxes(range=[0, size_data['中位数月薪'].max() * 1.15])
        st.plotly_chart(fig_size, use_container_width=True)
    #公司性质图，逻辑如上。
    with col6:
        st.subheader("不同公司性质的薪酬对比")
        fig_type = px.bar(type_data, x=type_data.index, y='中位数月薪', text='中位数月薪',
                          title='公司性质 vs. 薪酬中位数')
        # 增加左边距(l=120)，为Y轴标签留出足够空间
        fig_type.update_layout(title_x=0.5, xaxis_title=None, margin=dict(l=120))
        fig_type.update_traces(texttemplate='%{y:,.0f} 元', textposition='outside')
        # 为Y轴范围增加一些顶部空间
        if not type_data.empty:
            fig_type.update_yaxes(range=[0, type_data['中位数月薪'].max() * 1.15])
        st.plotly_chart(fig_type, use_container_width=True)

    # 第一级：核心概要
    st.markdown("核心结论：公司规模与薪酬水平整体呈正相关；在性质上，上市公司与外资企业提供了最具竞争力的薪酬。")

    # 第二级：可展开的详细解读
    with st.expander("点击查看详细解读 👇"):
        st.markdown("""
        *   **规模效应:** 从“少于15人”的初创团队，到“10000人以上”的巨型企业，薪酬中位数随着公司规模的扩大而稳步提升，清晰地证明了“大厂”在薪酬上的优势。
        *   **性质差异:** `上市公司` 和 `外资企业` 在薪酬中位数上处于第一梯队，是市场的“薪酬标杆”。`国有企业` 和 `合资企业` 提供了优厚且稳定的次级选择。而 `民营公司` 作为数量最庞大的市场主体，其薪酬中位数构成了整个行业的“基准线”。
        """)

    st.write("---")

    # --- 模块五：岗位热力图 --（可交互式图表，通过鼠标进行缩放等操作）
    st.header("5. 全国岗位密度热力图")
    # 选取“岗位发布-lat”（纬度）和“岗位发布-lon”（经度）这两列进行岗位热力图绘制
    map_data = df_display[['岗位发布-lat', '岗位发布-lon']].rename(
        columns={'岗位发布-lat': 'lat', '岗位发布-lon': 'lon'})
    # 筛选出所有经纬度都为正数（在中国境内）的、有效的地理坐标点。过滤掉了所有无效的地理位置数据，防止 pydeck 在渲染时因为遇到非法坐标而报错。
    map_data = map_data[(map_data['lat'] > 0) & (map_data['lon'] > 0)]
    #  st.pydeck_chart最终渲染，设置地图初始的中心点与视角，展示 pydeck 复杂3D地图。
    if not map_data.empty:
        st.pydeck_chart(pdk.Deck(layers=[pdk.Layer('HeatmapLayer', data=map_data, get_position='[lon, lat]')],
                                 initial_view_state=pdk.ViewState(latitude=36, longitude=104, zoom=3.5, pitch=45)))

else:
    # ---------------------------------
    #   模式二：“深度下钻” (当使用侧边筛选器筛选时)
    # ---------------------------------
    st.success(f"🔍 **您正在对【{len(df_display)}】个特定岗位进行【深度下钻】分析。**")

    # --- 下钻分析一：该群体的热门岗位 ---（交互式的、水平方向的条形图）
    st.header("1. 该群体的热门岗位类别")
    # 接收的df_display是已经经过侧边栏全局筛选后的结果。这意味着prepare_category_data 函数每次都是在正确的数据子集上进行计算的。
    # 将复杂的数据处理逻辑，完全封装在了 prepare_category_data 函数内部。使主体函数只负责“调用”和“展示”。
    hot_data, _ = prepare_category_data(df_display, cache_key=len(df_display))
    # 用“链式调用”，将如x轴柱子、数据标签、文本、背景、图表内部标签等配置写在一起。
    # 'total ascending'使图表Y轴每个类别相对应的X轴的值（也就是'岗位数量'），对Y轴的类别进行升序排列。
    fig_hot_drill = (px.bar(
        hot_data, x='岗位数量', y='检索二级职位类别', text='岗位数量',
        orientation='h', template='plotly_white', title='该群体热门岗位 TOP 15',
        labels={'检索二级职位类别': '职位类别'}
    ).update_layout(
        title_x=0.5, xaxis_title='岗位数量 (个)', yaxis_title="职位类别",
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=150)  # 增加左边距
    ).update_traces(
        textposition='outside'
    ))
    st.plotly_chart(fig_hot_drill, use_container_width=True)

    # --- 下钻分析二：该群体的薪资分布 ---（直方图、箱型图）
    st.header("2. 该群体的薪资分布")
    # 创造两列布局
    col_hist, col_box = st.columns(2)
    #df_display:使用全局筛选后的最终数据集.nbins=50:指定了要将整个薪资范围，划分成50个等宽的区间
    with col_hist:
        st.subheader("薪资分布直方图")
        fig_hist_drill = px.histogram(df_display, x="月薪", title="薪资分布", nbins=50)
        fig_hist_drill.update_layout(title_x=0.5)
        # 将创造好的直方图渲染到网页上.
        st.plotly_chart(fig_hist_drill, use_container_width=True)
    with col_box:
        st.subheader("薪资分布箱形图")
        # points="all"以半透明抖动点的方式，展示出所有的原始数据点,可直观地感受到数据的原始分布密度.
        fig_box_drill = px.box(df_display, y="月薪", points="all", title="薪资分布")
        # type="log":对Y轴进行非线性的“压缩”，特别是对高值部分.可以全面看到筛选后的数据原貌与内部结构(中位数、上下四分位数）.
        # use_container_width=True,动态调整将图表宽度调整为容器一样宽
        fig_box_drill.update_yaxes(type="log")
        fig_box_drill.update_layout(title_x=0.5)
        st.plotly_chart(fig_box_drill, use_container_width=True)
    st.markdown(
        "> **解读:** 直方图的“山峰”揭示了薪资的**主体**集中区间；而箱形图（已启用对数坐标轴）则更清晰地展示了**中位数**、**核心50%**的范围、以及**离群值**的分布情况。")

    st.write("---")

    # --- 下钻分析三：岗位热力图 ---
    st.header("3. 该群体的岗位地理分布")
    # df_display使用筛选器筛选过的地区, '检索城市'是为了后续实现智能变焦.
    map_data_drill = df_display[['岗位发布-lat', '岗位发布-lon', '检索城市']].copy()
    map_data_drill.rename(columns={'岗位发布-lat': 'lat', '岗位发布-lon': 'lon'}, inplace=True)
    map_data_drill = map_data_drill[(map_data_drill['lat'] > 0) & (map_data_drill['lon'] > 0)]

    if not map_data_drill.empty:

        # 获取用户筛选了多少个不同的城市,判断用户行为
        num_selected_cities = map_data_drill['检索城市'].nunique()

        # 根据城市数量，决定地图的中心点和缩放等级
        if num_selected_cities > 1:
            # 如果选择了多个(>1)城市，从聚焦单一城市视角到使用全中国的宏观视角,看到筛选出的城市
            initial_view_state_drill = pdk.ViewState(
                latitude=36,
                longitude=104,
                zoom=3.5,
                pitch=45
            )
        else:
            # 如果只选择了一个城市就聚焦于该城市的中心
            initial_view_state_drill = pdk.ViewState(
                latitude=map_data_drill['lat'].mean(),
                longitude=map_data_drill['lon'].mean(),
                zoom=9,
                pitch=50
            )

        # 3. 渲染地图
        # initial_view_state=initial_view_state_drill使用上面提到的逻辑动态切换视角
        heatmap_layer_drill = pdk.Layer('HeatmapLayer', data=map_data_drill, get_position='[lon, lat]', opacity=0.8)
        r_drill = pdk.Deck(layers=[heatmap_layer_drill], initial_view_state=initial_view_state_drill, map_style='light')
        st.pydeck_chart(r_drill)
    else:
        st.warning("该筛选条件下，无有效的地理位置数据。")

    st.write("---")

    # --- 下钻分析四：技能与福利画像 ---
    st.header("4. 岗位画像词云 (可二次下钻)")
    st.markdown("> **说明:** 您可以先选择一个**岗位角色**，再进一步选择一个**核心技术**，进行精准画像。")
    # 定义两个独立的选项字典,定义了只属于这个模块的“岗位角色”和“核心技术”选项,以便进行分析.
    ROLE_OPTIONS = {
        "👉 查看筛选群体的整体画像": None,
        "后端开发": "后端|Java|Python|Go|PHP|C++",
        "前端开发": "前端|Vue|React|Web",
        "人工智能": "算法|AI|机器学习|深度学习|NLP",
        "数据分析": "数据分析|BI|数据挖掘",
        "测试开发": "测试|测开|QA",
    }
    TECH_OPTIONS = {
        "👉 不限特定技术": None,
        "Java": "Java(?!Script)",
        "Python": "Python",
        "C++": "C\+\+",
        "Go": "Go语言|Golang",
    }

    # 创建两个独立的下拉选择器
    col_role, col_tech = st.columns(2)
    with col_role:
        selected_role = st.selectbox(
            label="第一步：请选择一个岗位角色",
            options=list(ROLE_OPTIONS.keys()),
            #当一个Streamlit应用中，有多个结构相同或标签相同的组件时，
            # 为它们提供一个唯一的key，可以帮助Streamlit准确地区分它们，避免状态混淆,防止与全局冲突.
            key='drilldown_role_select'
        )
    with col_tech:
        selected_tech = st.selectbox(
            label="第二步：请选择一个核心技术 (可选)",
            options=list(TECH_OPTIONS.keys()),
            key='drilldown_tech_select'  # 添加一个唯一的key,逻辑如上
        )

    # 双层过滤
    #df_for_wordcloud = df_display.copy()确保从筛选器筛选过后的副本中进行下钻操作.
    df_for_wordcloud = df_display.copy()
    title_profile = "整体"
    title_parts = []
    # 检查用户是否在“角色”下拉框中做出了有效选择。如果选择了，就用 str.contains() 在 岗位名列中进行模糊搜索，过滤无用数据.
    # title_parts.append(selected_role): 将选择的角色名，添加到一个列表中，为后续的动态标题做准备。
    if selected_role and ROLE_OPTIONS[selected_role]:
        df_for_wordcloud = df_for_wordcloud[
            df_for_wordcloud['岗位名'].str.contains(ROLE_OPTIONS[selected_role], case=False, na=False)
        ]
        title_parts.append(selected_role)
    # 在第一层过滤后确定用户是否选择核心技术.
    # 使用逻辑“或”(|)，在'岗位名'和'岗位描述'两列中同时搜索技术关键词，这大大提高了筛选的准确率(部分岗位要求写在岗位名中)
    if selected_tech and TECH_OPTIONS[selected_tech]:
        search_tech = TECH_OPTIONS[selected_tech]
        df_for_wordcloud = df_for_wordcloud[
            df_for_wordcloud['岗位名'].str.contains(search_tech, case=False, na=False) |
            df_for_wordcloud['岗位描述'].str.contains(search_tech, case=False, na=False)
            ]
        title_parts.append(selected_tech)
    # 动态生成标题。如果用户进行了任何局部筛选，title_parts 列表就不为空.使用join函数与&符号生成复合标题.
    if title_parts:
        title_profile = " & ".join(title_parts)
    #     防止在两栏中没有筛选到数据出现报错,中断程序
    if df_for_wordcloud.empty:
        st.warning(f"在当前筛选条件下，没有找到与“{title_profile}”相关的岗位。")
    else:
        col_skill, col_benefit = st.columns(2)
        with col_skill:
            st.subheader(f"{title_profile} - 核心技术画像")
            # 调用函数时，把动态生成的 title_profile 作为 cache_key 传进去
            # 确保了只有当任何一个筛选条件（全局或局部）发生变化，导致这个cache_key字符串唯一时，
            # generate_wordcloud_image 函数才会去执行耗时的重新计算。(调用提前缓存的数据)

            skill_cache_key = f"skill_{title_profile}_{len(df_for_wordcloud)}"
            # fig_skill = generate_wordcloud_image调用的是我们定义在A区域全局变量下的核心函数.
            # use_stopwords=True:为技术词云开启停用词过滤.
            fig_skill = generate_wordcloud_image(
                df_for_wordcloud,
                '岗位描述',
                use_stopwords=True,
                cache_key=skill_cache_key  # 例如 "skill_后端开发 & Java"
            )
            # 检测是否生成图像对象
            if fig_skill:
                st.pyplot(fig_skill)
            else:
                st.warning("无足够数据生成核心技术词云。")
        # 福利词云代码逻辑大致如上
        with col_benefit:
            st.subheader(f"{title_profile} - 福利待遇画像")

            benefit_cache_key = f"benefit_{title_profile}_{len(df_for_wordcloud)}"

            fig_benefit = generate_wordcloud_image(
                df_for_wordcloud,
                '岗位福利待遇',
                use_stopwords=False,
                cache_key=benefit_cache_key
            )
            if fig_benefit:
                st.pyplot(fig_benefit)
            else:
                st.warning("无足够数据生成福利待遇词云。")
    st.write("---")

    # --- 下钻分析五：数据详情浏览器(可以查看详细数据)
    st.header("5. 数据详情浏览器")
    with st.expander("点击展开/折叠，查看当前筛选条件下的具体岗位数据 👇"):
        st.dataframe(df_display[['岗位名', '公司名称', '月薪', '学历', '经验标签', '检索城市', '岗位福利待遇']])





