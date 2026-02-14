import pandas as pd
import os

print("--- 通用岗位类别内部审 ---")

final_file = "it_data_cleaned_v6_final.csv"
if not os.path.exists(final_file):
    print(f"   错误: 未找到最终数据集 '{final_file}'。")
else:
    print(f"\n一、加载V6最终版数据")
    df = pd.read_csv(final_file)
    print(f"   数据加载完成。")
    print("-" * 50)

    print("\n二、确定需要进行审查的目标岗位类别")
    hot_jobs = df.groupby('检索二级职位类别').size().reset_index(name='岗位数量')
    top_hot_categories = hot_jobs.sort_values(by='岗位数量', ascending=False).head(15)['检索二级职位类别'].tolist()
    high_salary_jobs = df.groupby('检索二级职位类别')['月薪'].agg(['median', 'size']).reset_index()
    high_salary_jobs.columns = ['检索二级职位类别', '中位数月薪', '岗位数量']
    significant_jobs = high_salary_jobs[high_salary_jobs['岗位数量'] >= 50]
    top_high_salary_categories = significant_jobs.sort_values(by='中位数月薪', ascending=False).head(15)[
        '检索二级职位类别'].tolist()
    all_target_categories = sorted(list(set(top_hot_categories + top_high_salary_categories)))
    print(f"   将对以下 {len(all_target_categories)} 个重要类别进行内部构成分析：")
    print(f"      {all_target_categories}")

    print("\n三、开始对各类别进行循环审查")

    for target_category in all_target_categories:
        print("\n" + "=" * 70)
        print(f"   审查类别: 【{target_category}】")
        print("=" * 70)
        category_df = df[df['检索二级职位类别'] == target_category].copy()


        if category_df.empty:
            print(f"   在数据集中未找到【{target_category}】类别。")
        else:
            print(f"   共找到 {len(category_df)} 个该类别的岗位。")
            job_title_counts = category_df['岗位名'].value_counts()
            print(f"\n      --- 【{target_category}】类别下，出现频率最高的10个具体岗位名 ---")
            print(job_title_counts.head(10))

print("\n 执行完毕 ---")
