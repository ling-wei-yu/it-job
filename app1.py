# ----------------------------------------------------------------------
# 毕业设计：基于Python的IT行业招聘数据可视化分析系统

# ----------------------------------------------------------------------

# --- 核心库导入 ---
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pydeck as pdk

# ======================================================================
#   (A) 页面基础设置、全局常量、核心函数定义
# ======================================================================
st.set_page_config(page_title="IT行业招聘数据分析系统", page_icon="💼", layout="wide")

# --- 全局常量 ---
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


# --- 数据加载与准备函数  ---
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path): st.error(f"错误: 未找到数据文件 '{file_path}'。"); return None
    df = pd.read_csv(file_path)
    df['城市等级'] = df['检索城市'].map(CITY_TIER_MAP).fillna('其他城市')
    return df


@st.cache_data
def prepare_experience_data(_df, mode='overall'):
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

        def remove_outliers(group):
            q1 = group['月薪'].quantile(0.25)
            q3 = group['月薪'].quantile(0.75)
            return group[(group['月薪'] >= q1) & (group['月薪'] <= q3)]

        bachelor_df_no_na = bachelor_df.dropna(subset=['经验等级'])
        source_df = bachelor_df_no_na.groupby('经验等级').apply(remove_outliers).reset_index(drop=True)
    else:  # 默认为 'overall'
        source_df = _df.copy()

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


@st.cache_data
def prepare_education_data(_df):
    """准备“学历价值分析”所需的数据。"""
    edu_to_analyze = ['中专/中技', '高中', '大专', '本科', '硕士', '博士']
    analysis_df = _df[_df['学历'].isin(edu_to_analyze)]
    edu_salary_median = analysis_df.groupby('学历')['月薪'].median().round(0).reset_index()
    ordered_edu_dtype = pd.CategoricalDtype(categories=EDUCATION_ORDER, ordered=True)
    edu_salary_median['学历'] = edu_salary_median['学历'].astype(ordered_edu_dtype)
    edu_salary_median = edu_salary_median.sort_values(by='学历')
    return edu_salary_median


@st.cache_data
def prepare_category_data(_df):
    """准备“岗位类别分析”所需的数据。"""
    hot_jobs = _df.groupby('检索二级职位类别').size().reset_index(name='岗位数量')
    top_15_hot = hot_jobs.sort_values(by='岗位数量', ascending=False).head(15)
    high_salary_jobs = _df.groupby('检索二级职位类别')['月薪'].agg(['median', 'size']).reset_index()
    high_salary_jobs.columns = ['检索二级职位类别', '中位数月薪', '岗位数量']
    significant_jobs = high_salary_jobs[high_salary_jobs['岗位数量'] >= 50]
    top_15_high = significant_jobs.sort_values(by='中位数月薪', ascending=False).head(15)
    return top_15_hot, top_15_high


def prepare_company_data(_df):
    # 准备“企业画像分析”所需的数据。

    # --- 公司规模分析 ---
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
def generate_wordcloud_image(_df, column_name, use_stopwords=False, cache_key="default"):
    if column_name not in _df.columns: return None
    text = " ".join(item for item in _df[column_name].dropna())
    if not text.strip(): return None

    word_list = jieba.cut(text)

    if use_stopwords:
        stopwords = {...}  # 您的停用词列表
        filtered_words = [word for word in word_list if len(word) > 1 and word not in stopwords]
    else:
        filtered_words = [word for word in word_list if len(word) > 1]

    if not filtered_words: return None

    # 检查字体文件
    font_path = 'simhei.ttf'
    if not os.path.exists(font_path):
        st.error(f"错误: 未找到中文字体 '{font_path}'。")
        return None

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

# --- 侧边栏筛选器 ---
st.sidebar.header("🔬 全局筛选器")
selected_cities = st.sidebar.multiselect('选择城市 (可多选)', options=sorted(df['检索城市'].unique()), default=[])
selected_education = st.sidebar.multiselect('选择学历 (可多选)', options=sorted(df['学历'].unique()), default=[])
selected_experience = st.sidebar.multiselect('选择工作经验 (可多选)', options=sorted(df['经验标签'].unique()),
                                             default=[])

df_filtered = df.copy()
if selected_cities: df_filtered = df_filtered[df_filtered['检索城市'].isin(selected_cities)]
if selected_education: df_filtered = df_filtered[df_filtered['学历'].isin(selected_education)]
if selected_experience: df_filtered = df_filtered[df_filtered['经验标签'].isin(selected_experience)]

st.sidebar.write("---")
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

    # --- 模块一：经验回报率 ---
    st.header("1. 经验回报率分析：多群体对比")
    tab1, tab2, tab3 = st.tabs(["📈 总体趋势分析", "🎓 本科生专属分析", "🎯 核心本科生分析"])
    with tab1:
        st.subheader("全量数据：平均值 vs. 中位数")
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

    # --- 模块二：学历价值 ---
    st.header("2. 学历价值分析")
    edu_data = prepare_education_data(df_display)
    fig_edu = px.bar(edu_data, x='学历', y='月薪', text='月薪', color='学历', template='plotly_white',
                     title="不同学历层次的IT岗位薪酬中位数对比")
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

    # --- 模块三：岗位类别 ---
    st.header("3. 市场热点分析：热门 vs. 高薪岗位类别")

    hot_data, high_salary_data = prepare_category_data(df_display)
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("热门岗位 TOP 15 (按需求量)")

        # 使用“链式调用”，完成所有配置
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

        # 使用“链式调用”
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

    # 模块四企业画像分析
    st.header("4. 企业画像分析：规模与性质的对比")
    size_data, type_data = prepare_company_data(df_filtered)

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("不同公司规模的薪酬对比")
        fig_size = px.bar(size_data, x=size_data.index, y='中位数月薪', text='中位数月薪',
                          title='公司规模 vs. 薪酬中位数')
        fig_size.update_layout(title_x=0.5, xaxis_title=None)
        fig_size.update_traces(texttemplate='%{y:,.0f} 元', textposition='outside')
        if not size_data.empty:
            fig_size.update_yaxes(range=[0, size_data['中位数月薪'].max() * 1.15])
        st.plotly_chart(fig_size, use_container_width=True)

    with col6:
        st.subheader("不同公司性质的薪酬对比")
        fig_type = px.bar(type_data, x=type_data.index, y='中位数月薪', text='中位数月薪',
                          title='公司性质 vs. 薪酬中位数')
        # 核心修正：增加左边距(l=120)，为Y轴标签留出足够空间
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

    # --- 模块五：岗位热力图 --
    st.header("5. 全国岗位密度热力图")
    map_data = df_display[['岗位发布-lat', '岗位发布-lon']].rename(
        columns={'岗位发布-lat': 'lat', '岗位发布-lon': 'lon'})
    map_data = map_data[(map_data['lat'] > 0) & (map_data['lon'] > 0)]
    if not map_data.empty:
        st.pydeck_chart(pdk.Deck(layers=[pdk.Layer('HeatmapLayer', data=map_data, get_position='[lon, lat]')],
                                 initial_view_state=pdk.ViewState(latitude=36, longitude=104, zoom=3.5, pitch=45)))

else:
    # ---------------------------------
    #   模式二：“深度下钻” (当有筛选时)
    # ---------------------------------
    st.success(f"🔍 **您正在对【{len(df_display)}】个特定岗位进行【深度下钻】分析。**")

    # --- 下钻分析一：该群体的热门岗位 ---
    st.header("1. 该群体的热门岗位类别")
    hot_data, _ = prepare_category_data(df_display)
    # 用“链式调用”，将所有配置写在一起
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

    # --- 下钻分析二：该群体的薪资分布 ---
    st.header("2. 该群体的薪资分布")
    col_hist, col_box = st.columns(2)
    with col_hist:
        st.subheader("薪资分布直方图")
        fig_hist_drill = px.histogram(df_display, x="月薪", title="薪资分布", nbins=50)
        fig_hist_drill.update_layout(title_x=0.5)
        st.plotly_chart(fig_hist_drill, use_container_width=True)
    with col_box:
        st.subheader("薪资分布箱形图")
        fig_box_drill = px.box(df_display, y="月薪", points="all", title="薪资分布")

        fig_box_drill.update_yaxes(type="log")
        fig_box_drill.update_layout(title_x=0.5)
        st.plotly_chart(fig_box_drill, use_container_width=True)
    st.markdown(
        "> **解读:** 直方图的“山峰”揭示了薪资的**主体**集中区间；而箱形图（已启用对数坐标轴）则更清晰地展示了**中位数**、**核心50%**的范围、以及**离群值**的分布情况。")

    st.write("---")

    # --- 下钻分析三：岗位热力图 ---
    st.header("3. 该群体的岗位地理分布")

    map_data_drill = df_display[['岗位发布-lat', '岗位发布-lon', '检索城市']].copy()
    map_data_drill.rename(columns={'岗位发布-lat': 'lat', '岗位发布-lon': 'lon'}, inplace=True)
    map_data_drill = map_data_drill[(map_data_drill['lat'] > 0) & (map_data_drill['lon'] > 0)]

    if not map_data_drill.empty:
        #

        # 1. 获取用户筛选了多少个不同的城市
        num_selected_cities = map_data_drill['检索城市'].nunique()

        # 2. 根据城市数量，决定地图的中心点和缩放等级
        if num_selected_cities > 1:
            # 如果选择了多个城市，使用全中国的宏观视角
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
        heatmap_layer_drill = pdk.Layer('HeatmapLayer', data=map_data_drill, get_position='[lon, lat]', opacity=0.8)
        r_drill = pdk.Deck(layers=[heatmap_layer_drill], initial_view_state=initial_view_state_drill, map_style='light')
        st.pydeck_chart(r_drill)
    else:
        st.warning("该筛选条件下，无有效的地理位置数据。")

    st.write("---")

    # --- 下钻分析四：技能与福利画像 ---
    st.header("4. 岗位画像词云 (可二次下钻)")
    st.markdown("> **说明:** 您可以先选择一个**岗位角色**，再进一步选择一个**核心技术**，进行精准画像。")

# --- 下钻分析五：数据详情浏览器 ---
st.header("5. 数据详情浏览器")
with st.expander("点击展开/折叠，查看当前筛选条件下的具体岗位数据 👇"):
    st.dataframe(df_display[['岗位名', '公司名称', '月薪', '学历', '经验标签', '检索城市', '岗位福利待遇']])
