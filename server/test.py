import os
import json
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import Get_file, scearch_content, Visualization

def create_test_data():
    """创建测试数据"""
    # 创建临时目录和测试文件
    test_dir = "E:\\langurage_process\\pro\\test"
    
    # 确保测试目录存在
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建测试文本文件
    test_files = {
        "test1.txt": "这是一个测试文档。包含中文内容，用于测试文本处理功能。人工智能技术发展迅速。",
        "test2.txt": "另一个测试文件。机器学习是人工智能的重要分支。深度学习技术应用广泛。",
        "test3.json": {"title": "JSON测试", "content": "这是JSON格式的测试数据", "keywords": ["测试", "数据", "格式"]},
        "test4.csv": "姓名,年龄,职业\n张三,25,工程师\n李四,30,设计师\n王五,28,产品经理"
    }
    
    for filename, content in test_files.items():
        filepath = os.path.join(test_dir, filename)
        if filename.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
    
    # 创建配置文件
    config_data = {"json_path": test_dir}
    config_path = os.path.join(test_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    return config_path, test_dir

def test_get_file_class():
    """测试 Get_file 类的功能"""
    print("=" * 50)
    print("测试 Get_file 类")
    print("=" * 50)
    
    try:
        config_path, test_dir = create_test_data()
        print(f"✓ 测试数据创建成功: {test_dir}")
        
        # 初始化 Get_file 实例
        file_reader = Get_file(config_path)
        print("✓ Get_file 实例创建成功")
        
        # 测试文件读取
        file_reader.read_dir_file()
        print(f"✓ 文件读取完成，共读取 {len(file_reader.processed_files_content)} 个文件")
        
        # 显示读取的文件信息
        for file_path, content in file_reader.processed_files_content.items():
            filename = os.path.basename(file_path)
            if content.startswith("错误:"):
                print(f"  ✗ {filename}: {content}")
            else:
                print(f"  ✓ {filename}: 内容长度 {len(content)} 字符")
        
        # 测试词频统计
        word_dict = file_reader.all_word_count()
        print(f"✓ 词频统计完成，共统计 {len(word_dict)} 个词汇")
        print("  前10个高频词:", list(word_dict.items())[:10])
        
        # 测试倒排索引构建
        file_reader.build_inverted_index()
        print(f"✓ 倒排索引构建完成，共索引 {len(file_reader.inverted_index)} 个词汇")
        print(f"✓ 文档总数: {file_reader.total_docs}")
        
        return file_reader
        
    except Exception as e:
        print(f"✗ Get_file 测试失败: {str(e)}")
        return None

def test_search_content_class(file_reader):
    """测试 scearch_content 类的功能"""
    print("\n" + "=" * 50)
    print("测试 scearch_content 类")
    print("=" * 50)
    
    if not file_reader:
        print("✗ 无法进行搜索测试，文件读取器为空")
        return
    
    try:
        search_engine = scearch_content()
        print("✓ scearch_content 实例创建成功")
        
        # 测试查询预处理
        test_queries = ["人工智能", "测试数据", "机器学习技术"]
        
        for query in test_queries:
            print(f"\n--- 测试查询: '{query}' ---")
            
            # 预处理查询
            processed_query = search_engine.preprocess_query(query)
            print(f"  预处理结果: {processed_query}")
            
            # 测试 TF 搜索
            tf_results = search_engine.tf_search(
                query, 
                file_reader.inverted_index,
                file_reader.processed_files_content,
                file_reader.file_word_count
            )
            print(f"  ✓ TF搜索: 找到 {tf_results['total_results']} 个结果")
            
            # 测试 TF-IDF 搜索
            tfidf_results = search_engine.tfidf_search(
                query,
                file_reader.inverted_index,
                file_reader.processed_files_content,
                file_reader.file_word_count,
                file_reader.content_list
            )
            print(f"  ✓ TF-IDF搜索: 找到 {tfidf_results['total_results']} 个结果")
            
            # 测试词向量搜索
            vector_results = search_engine.word_vector_search(
                query,
                file_reader.inverted_index,
                file_reader.processed_files_content,
                file_reader.file_word_count,
                file_reader.content_list
            )
            print(f"  ✓ 词向量搜索: 找到 {vector_results['total_results']} 个结果")
            
            # 显示部分搜索结果
            if tf_results['found'] and tf_results['results']:
                result = tf_results['results'][0]
                print(f"    示例结果: {os.path.basename(result['file_name'])}")
                print(f"    匹配词: {result['matched_word']}")
                print(f"    TF分数: {result['tf_score']}")
        
        return search_engine
        
    except Exception as e:
        print(f"✗ scearch_content 测试失败: {str(e)}")
        return None

def test_visualization_class(file_reader):
    """测试 Visualization 类的功能"""
    print("\n" + "=" * 50)
    print("测试 Visualization 类")
    print("=" * 50)
    
    if not file_reader:
        print("✗ 无法进行可视化测试，文件读取器为空")
        return
    
    try:
        visualizer = Visualization()
        print("✓ Visualization 实例创建成功")
        print(f"✓ 输出目录: {visualizer.output_dir}")
        
        # 测试词云图生成
        print("\n--- 测试词云图生成 ---")
        wordcloud_result = visualizer.generate_wordcloud(
            "人工智能",
            file_reader.processed_files_content
        )
        if wordcloud_result['success']:
            print(f"  ✓ 词云图生成成功: {wordcloud_result['save_path']}")
            print(f"  ✓ 相关文档数: {wordcloud_result['relevant_docs']}")
        else:
            print(f"  ✗ 词云图生成失败: {wordcloud_result['message']}")
        
        # 测试关键词排行图
        print("\n--- 测试关键词排行图 ---")
        ranking_result = visualizer.generate_keyword_ranking(
            "测试",
            file_reader.processed_files_content,
            file_reader.file_word_count
        )
        if ranking_result['success']:
            print(f"  ✓ 关键词排行图生成成功: {ranking_result['save_path']}")
            print(f"  ✓ 关键词数量: {len(ranking_result['keywords'])}")
        else:
            print(f"  ✗ 关键词排行图生成失败: {ranking_result['message']}")
        
        # 测试文件扩展名分析
        print("\n--- 测试文件扩展名分析 ---")
        extension_result = visualizer.analyze_file_extensions(file_reader.file_name)
        if extension_result['success']:
            print(f"  ✓ 文件扩展名分析成功: {extension_result['save_path']}")
            print(f"  ✓ 文件类型统计: {extension_result['extension_stats']}")
        else:
            print(f"  ✗ 文件扩展名分析失败: {extension_result['message']}")
        
        # 测试词频图表
        print("\n--- 测试词频图表 ---")
        for chart_type in ['bar', 'line', 'pie']:
            chart_result = visualizer.create_word_frequency_chart(
                file_reader.word_dict,
                chart_type=chart_type,
                top_n=10
            )
            if chart_result['success']:
                print(f"  ✓ {chart_type.upper()}图生成成功: {chart_result['save_path']}")
            else:
                print(f"  ✗ {chart_type.upper()}图生成失败: {chart_result['message']}")
        
        return visualizer
        
    except Exception as e:
        print(f"✗ Visualization 测试失败: {str(e)}")
        return None

def test_integration():
    """集成测试"""
    print("\n" + "=" * 50)
    print("集成测试")
    print("=" * 50)
    
    try:
        # 完整流程测试
        config_path, test_dir = create_test_data()
        
        # 初始化并处理文件
        file_reader = Get_file(config_path)
        file_reader.init_read_dir()  # 完整初始化流程
        
        print(f"✓ 完整初始化成功")
        print(f"  - 处理文件数: {len(file_reader.processed_files_content)}")
        print(f"  - 词汇总数: {len(file_reader.word_dict)}")
        print(f"  - 索引词汇数: {len(file_reader.inverted_index)}")
        
        # 搜索测试
        search_engine = scearch_content()
        query = "人工智能技术"
        
        results = search_engine.tf_search(
            query,
            file_reader.inverted_index,
            file_reader.processed_files_content,
            file_reader.file_word_count
        )
        
        print(f"✓ 搜索测试完成: 查询'{query}'找到{results['total_results']}个结果")
        
        # 可视化测试
        visualizer = Visualization()
        vis_result = visualizer.generate_wordcloud(query, file_reader.processed_files_content)
        
        if vis_result['success']:
            print(f"✓ 可视化测试完成: 生成文件 {vis_result['save_path']}")
        
        print("\n✓ 所有集成测试通过!")
        
    except Exception as e:
        print(f"✗ 集成测试失败: {str(e)}")

def main():
    """主测试函数"""
    print("开始 utils.py 非网络功能测试")
    print("测试目录: E:\\langurage_process\\pro\\test")
    
    # 逐个测试各个模块
    file_reader = test_get_file_class()
    search_engine = test_search_content_class(file_reader)
    visualizer = test_visualization_class(file_reader)
    
    # 集成测试
    test_integration()
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)
    
    # 清理测试文件（可选）
    cleanup = input("\n是否清理测试文件? (y/n): ")
    if cleanup.lower() == 'y':
        import shutil
        test_dir = "E:\\langurage_process\\pro\\test"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print("✓ 测试文件已清理")

if __name__ == "__main__":
    main()
