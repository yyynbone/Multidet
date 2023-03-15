import os
import cv2
if __name__ == '__main__':
    save_dir = 'results/val/merge/zjdet_neck_exp_best'
    orig = 'exp3'
    comp = 'exp4'
    o_filess = []
    comp_filess = []
    sub_dir_name = ['not_matched', 'false_pred', 'matched']
    for is_igmatch in sub_dir_name:
        o_dir = os.path.join(save_dir, orig, 'visual_images', is_igmatch)
        comp_dir = os.path.join(save_dir, comp, 'visual_images', is_igmatch)
        o_filess.append(os.listdir(o_dir))
        comp_filess.append(os.listdir(comp_dir))
    for i, o_files in enumerate(o_filess):
        for o_file in o_files:
            if o_file not in comp_filess[i]:
                file_path = os.path.join(os.path.join(save_dir, orig, 'visual_images', sub_dir_name[i]),
                                         o_file)
                img =  cv2.imread(file_path)
                cv2.imshow(f'{sub_dir_name[i]}_{o_file}',img)
                for j in range(len(sub_dir_name)):
                    if j==i:
                        continue

                    if o_file in comp_filess[j]:
                        cfile_path = os.path.join(os.path.join(save_dir, comp, 'visual_images', sub_dir_name[j]),
                                                 o_file)
                        cimg = cv2.imread(cfile_path)
                        cv2.imshow(f'{sub_dir_name[j]}_{o_file}', cimg)
                    else:
                        print(f'{sub_dir_name[i]}_{o_file}  not existed in comp_file')
                cv2.waitKey(0)