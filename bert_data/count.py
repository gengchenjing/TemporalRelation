with open("mat_sdp.txt",'r+')as f1, open("mat_sdp4.txt",'w+')as f2, open("mat_sdp6.txt",'w+')as f3:
    for lines in f1.readlines():
        lines = str(lines)
        print(lines.replace("met-by", "after").replace("overlapped-by", "overlap").replace("finishes", "overlap").replace("during","overlap").replace(
                "started-by", "overlap").replace("equal", "overlap").replace("starts", "overlap").replace("contains","overlap").replace(
                "finished-by", "overlap").replace("overlaps", "overlap").replace("meets", "before").replace("is_included","overlap").replace(
                "identity", "overlap").replace("includes", "overlap").strip(), file=f2)
        print(lines.replace("met-by","after").replace("overlapped-by","oa").replace("finishes","oa").replace("during","overlap").replace("started-by","overlap").replace("equal","overlap").replace("starts","ob").replace("contains","overlap").replace("finished-by","overlap").replace("overlaps","ob").replace("meets","before").replace("is_included","overlap").replace("identity","overlap").replace("includes","overlap").strip(),file=f3)

        
