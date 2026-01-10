# This changes a bunch of directories to be owned by vscode because this is a docker container and actually
# we just want allow the vscode user to install python packages without having to worry about permissions.

SECONDS=0

echo "Be patient, this can take minutes the first time it's run"
sudo chown vscode /home/vscode/.cache
sudo chown -R vscode /usr/local/share &
sudo chown -R vscode /usr/local/lib &
sudo chown -R vscode /usr/local/etc &
sudo chown -R vscode /usr/local/src &
sudo chown -R vscode /usr/local/bin &
sudo chown -R vscode /usr/local/include &

# Wait for the background processes to complete
wait

# Calculate and display the elapsed time
elapsed_time=$SECONDS
echo "Total execution time: $((elapsed_time / 60)) minutes and $((elapsed_time % 60)) seconds."