function v3d2tif(path2dir)
	outdir = fullfile(path2dir, 'tif');
	if exist(outdir, 'dir')
        rmdir(outdir)
    else
    	mkdir(outdir)
	end

	v3drawlist = dir(fullfile(path2dir, '*.v3draw'));
	for i = 1 : numel(v3drawlist)
		fprintf('Converting %d/%d', i, numel(v3drawlist))
		v3dvol = load_v3d_raw_img_file(fullfile(path2dir, v3drawlist(i).name));
		outputFileName = fullfile(path2dir, 'tif', strcat(v3drawlist(i).name, '.tif'));

		for z = 1 : size(v3dvol, 3)
		   imwrite(v3dvol(:, :, z), outputFileName, 'WriteMode', 'append');
		end
	end
end