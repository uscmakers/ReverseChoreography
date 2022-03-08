function generate_plot(song, artist)
% ex) generate_plot(1, 0, "humble", "kendrick_lamar")

    for dancer_index = 1:3
        for run = 0:2

            fname = "preprocessed/"+dancer_index+"_"+run+"_"+song+"_"+artist+"_XDATA.csv";
            
            table = readtable(fname);
            
            node_labels = table.Properties.VariableNames;
            fs = 10;
            dt = 1/fs;
            
            fig = figure();
            fig.WindowState = 'maximized';
            index = 1;
            
            for i = 2:length(node_labels)
                temp_node = node_labels(i);
                temp_node = temp_node{1};
                y = table2array(table(:, temp_node));
                x = table2array(table(:, "Var1"))*dt;
               
                %figure;
                
                subplot(2, 9, index);
                plot(x, y);
                xlabel("Seconds");
                index = index + 1;
                subtitle(temp_node);
                
                %subplot(1, 3, 2);
                %spectrogram(x, y);
                %subtitle("Spectrogram");
            
                velocity = zeros(1, length(y));
                for j = 2:length(y)
                    dy = y(j) - y(j-1);
                    dx = dt;
                    velocity(1, j) = (dy/dx);
                end
            
                subplot(2, 9, index);
                plot(x, velocity);
                subtitle("Velocity");
                xlabel("Seconds");
                index = index + 1;
            
                %figure;
                %freqz(x, y);
                %title(temp_node);
            
            end
            
            sgtitle(song+" : "+artist+" Dancer "+dancer_index+" Run "+run); 
            
        end
    end

end