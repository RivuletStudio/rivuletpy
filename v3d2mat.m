function v3d2mat(path2dir)
	outdir = fullfile(path2dir, 'mat');
	if exist(outdir, 'dir')
        rmdir(outdir, 's')
    	mkdir(outdir)
    else
    	mkdir(outdir)
	end

	v3drawlist = dir(fullfile(path2dir, '*.v3draw'));
	for i = 1 : numel(v3drawlist)
		fprintf('Converting %d/%d', i, numel(v3drawlist))
		img = load_v3d_raw_img_file(fullfile(path2dir, v3drawlist(i).name));
		outputFileName = fullfile(path2dir, 'mat', strcat(v3drawlist(i).name, '.mat'));
		save(outputFileName, 'img')
	end
	disp('== Done ==')
end
