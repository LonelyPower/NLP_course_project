if __name__ == '__main__':
    import os
    import splitfolders as spf
    workplace_root = '/home/drew/Desktop/nlp_course_project/news_cls_recm'

    spf.ratio(
        input=os.path.join(workplace_root,'THUCNews'), 
        output=os.path.join(workplace_root,'data'), 
        seed=1145, 
        ratio=(0.8, 0.1, 0.1)
    )