#=== Deal with inputs.
if ARGV.length < 5
	puts "saps_wrapper.rb is a wrapper for the SAPS algorithm."
	puts "Usage: ruby saps_wrapper.rb <instance_relname> <instance_specifics> <cutoff_time> <cutoff_length> <seed> <params to be passed on>."
	exit -1
end
cnf_filename = ARGV[0]
instance_specifics = ARGV[1]
cutoff_time = ARGV[2].to_f
cutoff_length = ARGV[3].to_i
seed = ARGV[4].to_i

#=== Here I assume instance_specifics only contains the desired target quality or nothing at all for the instance, but it could contain more (to be specified in the instance_file or instance_seed_file)
if instance_specifics == ""
	qual = 0
else
	qual = instance_specifics.split[0]
end

paramstring = ARGV[5...ARGV.length].join(" ")
params = ARGV[5...ARGV.length].join("_")

#=== Build algorithm command and execute it.
i1 = cnf_filename.split("-")[0];
i2 = cnf_filename.split("-")[1];
cmd = "ACO.exe -i1 #{i1} -i2 #{i2} #{paramstring} -seed #{seed} -out 0 -core 4"

filename = "aco-log/#{params}_#{cnf_filename}_#{rand}.out"
exec_cmd = "#{cmd} > #{filename}"

puts "Calling: #{exec_cmd}"
system exec_cmd

#=== Parse algorithm output to extract relevant information for ParamILS.
runtime = nil
runlength = nil

File.open(filename){|file|
	while line = file.gets
		if line =~ /result = (\d+)$/
			runlength = $1.to_i
		end
	end
}
#File.delete(filename)
puts "Result for ParamILS: #{1}, #{1}, #{runlength}, #{1}, #{seed}"
