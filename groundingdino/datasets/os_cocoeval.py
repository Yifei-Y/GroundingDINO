import numpy as np
from collections import defaultdict
from pycocotools.cocoeval import COCOeval


class ClsAgnCOCOEval(COCOeval):
    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            if gt['ignore'] == 0:
                gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            # gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {} 

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((18,))
            stats[0]  = _summarize(1,  maxDets=self.params.maxDets[4])
            stats[1]  = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[4])
            stats[2]  = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[4])
            stats[3]  = _summarize(1, areaRng='small', maxDets=self.params.maxDets[4])
            stats[4]  = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[4])
            stats[5]  = _summarize(1, areaRng='large', maxDets=self.params.maxDets[4])
            stats[6]  = _summarize(0, maxDets=self.params.maxDets[0]) # 10
            stats[7]  = _summarize(0, maxDets=self.params.maxDets[1]) # 20
            stats[8]  = _summarize(0, maxDets=self.params.maxDets[2]) # 30
            stats[9]  = _summarize(0, maxDets=self.params.maxDets[3]) # 50
            stats[10] = _summarize(0, maxDets=self.params.maxDets[4]) # 100
            stats[11] = _summarize(0, maxDets=self.params.maxDets[5]) # 200
            stats[12] = _summarize(0, maxDets=self.params.maxDets[6]) # 300
            stats[13] = _summarize(0, maxDets=self.params.maxDets[7]) # 500
            stats[14] = _summarize(0, maxDets=self.params.maxDets[8]) # 1000
            stats[15] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[4])
            stats[16] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[4])
            stats[17] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[4])
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        self.stats = summarize()